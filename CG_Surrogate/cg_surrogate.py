import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class CG_Surrogate:
    def __init__(self, rom):
        self.rom = rom
        self.rom_autograd = self.rom.get_batched_autograd_fun()
        self.params = {'theta_c': None, 'tau_c': torch.ones(self.rom.mesh.n_cells, requires_grad=True),
                       'tau_cf': None}

        # for numerical stability (of, e.g., log)
        self.eps = 1e-12

    def train(self, designMatrix, output_data):
        """
        Trains the CG surrogate model.
        :param data: a DarcyData object
        :return:
        """
        self.params['theta_c'] = torch.zeros(designMatrix.shape[2], requires_grad=True)
        self.params['tau_cf'] = torch.ones(output_data.shape[1], requires_grad=True)
        log_lambda_c_init = torch.zeros(designMatrix.shape[0], designMatrix.shape[1], requires_grad=True)
        log_lambda_c_pred = designMatrix @ self.params['theta_c']
        log_lambda_c = LogLambdacParam(output_data, log_lambda_c_pred, init_value=log_lambda_c_init)
        log_lambda_c.optimizer = optim.Adam([log_lambda_c.value], lr=3e-3)
        log_lambda_c.scheduler = optim.lr_scheduler.ReduceLROnPlateau(log_lambda_c.optimizer, factor=.1,
                                                                      verbose=True, min_lr=1e-12, patience=15)
        log_lambda_c.converge(model=self, n_samples_tot=designMatrix.shape[0])

    def e_step(self):
        pass

    def loss_lambda_c(self, log_lambda_c, designMatrix, output_data):
        """
        For delta approximation of q(lambda_c)
        :param designMatrix:
        :param output_data:
        :return:
        """
        output_pred = self.rom_autograd(torch.exp(log_lambda_c))
        loss_p_cf = .5*(self.params['tau_cf']*(output_pred - output_data)**2).sum()
        loss_p_c = .5*(self.params['tau_c']*(log_lambda_c - designMatrix @ self.params['theta_c'])**2).sum()
        return loss_p_cf + loss_p_c

class DesignMatrix:
    def __init__(self):
        self.matrix = None       # first index for sample, second for cell, third for feature
        self.features = [lambda x: 1.0, torch.mean]       # list of feature functions
        self.mask = None

    def get_masks(self, fine_mesh, coarse_mesh):
        """
        Computes masks to extract fine scale pixels belonging to a ROM macro-cell
        :return:
        """
        self.mask = torch.zeros(coarse_mesh.n_cells, fine_mesh.n_cells, dtype=torch.bool)
        for c, fine_cell in enumerate(fine_mesh.cells):
            for r, coarse_cell in enumerate(coarse_mesh.cells):
                if coarse_cell.is_inside(fine_cell.centroid.unsqueeze(0)):
                    self.mask[r, c] = True

    def assemble(self, input_data, dim_rom):
        """
        Computes the design matrix
        :param data: data object for which the design matrix needs to be computed
        :return:
        """
        self.matrix = torch.zeros(len(input_data), dim_rom, len(self.features))

        for n, permeability_sample in enumerate(input_data):
            for k in range(dim_rom):
                for j, feature in enumerate(self.features):
                    self.matrix[n, k, j] = feature(permeability_sample[self.mask[k]])


class ModelParam:
    """
    Class for a certain parameter of the model, i.e., thetaf, thetac, log_lambdac_mean, and z_mean
    """
    def __init__(self, init_value=None, batch_size=128):
        self.value = init_value                                 # Torch tensor of parameter values
        self.lr_init = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]     # Initial learning rate in every convergence iteration
        self.optimizer = None                                   # needs to be specified later
        self.scheduler = None
        self.batch_size = batch_size                            # the batch size for optimization of the parameter
        self.convergence_iteration = 0                          # iterations with full convergence
        self.step_iteration = 0                                 # iterations in the current optimization
        self.max_iter = 100
        self.lr_drop_factor = 3e-4                              # lr can drop by this factor until convergence
        self.eps = 1e-12                                        # for stability of log etc.

    def loss(self, **kwargs):
        """
        The loss function pertaining to the parameter
        Override this!
        """
        raise Exception('loss function not implemented')

    def step(self, **kwargs):
        """
        Performs a single optimization iteration
        Override this!
        """
        raise Exception('step function not implemented')

    def draw_batch_samples(self, n_samples_tot, mode='shuffle'):
        """
        Draws a batch sample. n_samples_tot is the size of the data set
        """
        if mode == 'shuffle':
            while True:
                yield torch.multinomial(torch.ones(n_samples_tot), self.batch_size)
        elif mode == 'iter':
            k = 0
            while True:
                k += 1
                yield torch.tensor(range(k*self.batch_size, (k + 1)*self.batch_size)) % n_samples_tot
        elif isinstance(mode, int):
            while True:
                yield mode

    def converge(self, model, n_samples_tot, mode='shuffle'):
        """
        Runs optimization of parameter till a convergence criterion is met
        """
        self.optimizer.param_groups[0]['lr'] = self.lr_init[min(self.convergence_iteration, len(self.lr_init) - 1)]
        batch_gen = self.draw_batch_samples(n_samples_tot, mode)
        while self.step_iteration < self.max_iter and self.optimizer.param_groups[0]['lr'] > \
                self.lr_drop_factor * self.lr_init[min(self.convergence_iteration, len(self.lr_init) - 1)]:
            batch_samples = next(batch_gen)
            self.step(model, batch_samples)
            self.step_iteration += 1
        self.convergence_iteration += 1
        self.step_iteration = 0


class LogLambdacParam(ModelParam):
    """
    Class pertaining to the log_lambdac_mean parameters
    """

    def __init__(self, output_data, log_lambdac_pred, init_value=None):
        super().__init__(init_value=init_value, batch_size=1)
        self.output_data = output_data
        self.log_lambdac_pred = log_lambdac_pred        # needs to be set before every convergence iteration

    def loss(self, model, output_data):
        """
        For delta approximation of q(lambda_c)
        :param designMatrix:
        :param output_data:
        :return:
        """
        output_pred = model.rom_autograd(torch.exp(self.value))
        loss_p_cf = .5 * (model.params['tau_cf'] * (output_pred - output_data) ** 2).sum()
        loss_p_c = .5 * (model.params['tau_c'] * (self.value - self.log_lambdac_pred) ** 2).sum()
        return loss_p_cf + loss_p_c

    def step(self, model, batch_samples):
        """
        One training step for log_lambdac_mean
        Needs to be updated to samples of lambda_c from approximate posterior
        model:  GenerativeSurrogate object
        """
        self.optimizer.zero_grad()
        uf_pred = model.rom_autograd(torch.exp(self.value) + self.eps)
        assert torch.all(torch.isfinite(uf_pred))
        loss = self.loss(model, uf_pred)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
        if (self.step_iteration % 500) == 0:
            print('loss_lambda_c = ', loss.item())

        # if __debug__:
        #     # print('loss_lambdac = ', loss)
        #     self.writer.add_scalar('Loss/train_pcf', loss)
        #     self.writer.close()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(loss)





