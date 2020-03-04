import socket
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
from poisson_fem import PoissonFEM
import rom_inverse as ri
import pyro.contrib.gp as gp
import ROM
import os




'''File for Stokes and Darcy training/test data'''


class Data:
    # Base class for Stokes/Darcy training/testing data
    def __init__(self, supervised_samples, unsupervised_samples, dtype=torch.float):
        assert len(supervised_samples.intersection(unsupervised_samples)) == 0, "samples both supervised/" \
                                                                                "unsupervised"
        self.dtype = dtype
        self.resolution = [128, 128]               # mesh resolution
        self.supervised_samples = supervised_samples              # list of sample numbers
        self.n_supervised_samples = len(supervised_samples)
        self.unsupervised_samples = unsupervised_samples
        self.n_unsupervised_samples = len(unsupervised_samples)
        self.path = ''                    # path to input data folder
        self.solution_folder = ''
        self.output = []                  # response field, i.e., output data
        self.input = []


class DarcyData(Data):
    def __init__(self, perm_length_scale=torch.tensor([.1, .1]), perm_var=torch.tensor(1.), nugget=1e-3,
                 supervised_samples=set(range(16)), unsupervised_samples=set()):

        super().__init__(supervised_samples, unsupervised_samples)
        permeability_covariance_function = gp.kernels.RBF(input_dim=2, variance=perm_var, lengthscale=perm_length_scale)
        self.permeability_random_field = ri.DiscretizedRandomField(self.resolution,
                                                                   kernel=permeability_covariance_function,
                                                                   nugget=nugget)

        self.bc_coefficients = torch.tensor([1., 1., 0.])       # Boundary condition function coefficients
        self.mesh = None
        self.rhs = None
        self.stiffness_matrix = None
        self.solver = None

        self.set_path_name()

    def set_path_name(self):
        assert len(self.path) == 0, 'Data path already set'
        self.path += '/home/constantin/'

        if socket.gethostname() == 'workstation1-room0436':
            # office workstation
            self.path += 'cluster/'

        self.path += f'python/data/rom_inverse/darcy_fom/resolution={self.resolution}/logGP/' \
                     f'kernel={self.permeability_random_field.kernel}/'
        for kernel_param in self.permeability_random_field.kernel.named_pyro_params():
            self.path += f'{kernel_param[0]}='
            if torch.is_tensor(kernel_param[1]):
                if len(kernel_param[1].shape) > 0:
                    for val in kernel_param[1]:
                        self.path += '%.2f_' % val.item()
                    # remove last underscore
                    pth = list(self.path)
                    pth[-1] = ''
                    self.path = ''.join(pth)
                else:
                    self.path += '%.2f' % kernel_param[1].item()
            self.path += '/'
        self.path += '/bc='
        for bc_coeff in self.bc_coefficients:
            self.path += '%.2f_' % bc_coeff.item()
        # remove last underscore
        pth = list(self.path)
        pth[-1] = ''
        self.path = ''.join(pth) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @staticmethod
    def origin(x):
        return torch.abs(x[0]) < torch.finfo(torch.float32).eps and torch.abs(x[1]) < torch.finfo(torch.float32).eps

    def ess_boundary_fun(self, x):
        return self.bc_coefficients[0]*x[0] + self.bc_coefficients[1]*x[1] + self.bc_coefficients[2]*x[0]*x[1]

    @staticmethod
    def domain_boundary(x):
        # unit square
        return torch.abs(x[0]) < torch.finfo(torch.float32).eps or torch.abs(x[1]) < torch.finfo(torch.float32).eps or \
               torch.abs(x[0]) > 1.0 - torch.finfo(torch.float32).eps or torch.abs(x[1]) > 1.0 - torch.finfo(
            torch.float32).eps

    def flux_field(self, x):
        q = np.array([self.bc_coefficients[0] + self.bc_coefficients[2]*x[1],
                      self.bc_coefficients[1] + self.bc_coefficients[2]*x[0]])
        return q

    def set_mesh(self):
        self.mesh = PoissonFEM.RectangularMesh(grid_x=torch.ones(self.resolution[0])/self.resolution[0],
                                               grid_y=torch.ones(self.resolution[1])/self.resolution[1])
        self.mesh.set_essential_boundary(self.origin, self.ess_boundary_fun)
        self.mesh.set_natural_boundary(self.domain_boundary)

    def set_rhs_and_stiffness(self):
        # Specify right hand side and stiffness matrix
        self.rhs = PoissonFEM.RightHandSide(self.mesh)
        self.rhs.set_natural_rhs(self.mesh, self.flux_field)
        self.stiffness_matrix = PoissonFEM.StiffnessMatrix(self.mesh)
        self.rhs.set_rhs_stencil(self.mesh, self.stiffness_matrix)

    def set_solver(self):
        # define fom
        FOM = ROM.ROM(self.mesh, self.stiffness_matrix, self.rhs, self.resolution[0]*self.resolution[1])
        # Change for non unit square domains!!
        # xx, yy = torch.meshgrid([torch.linspace(0, 1, self.resolution[0] + 1),
        #                          torch.linspace(0, 1, self.resolution[1] + 1)])
        # X = torch.cat((xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)), 1)
        # FOM.mesh.get_interpolation_matrix(X)

        def solver(lmbda):
            lmbda = lmbda.detach().numpy()
            # for stability
            lmbda[lmbda > 1e12] = 1e12
            FOM.solve(lmbda)

            # scatter essential boundary conditions
            FOM.set_full_solution()
            return torch.tensor(FOM.full_solution.array, dtype=FOM.dtype)
        self.solver = solver

    def gen_data(self):
        """
        Generates data and saves it to disk, not memory!
        :return:
        """
        for n in self.supervised_samples:
            permeability_sample = self.permeability_random_field.sample_permeability()
            solution = self.solver(permeability_sample)
            torch.save(permeability_sample, self.path + f'permeability{n}.pt')
            torch.save(solution, self.path + f'solution{n}.pt')

    def load_input_data(self):
        pass



class StokesData(Data):
    def __init__(self, supervised_samples=set(), unsupervised_samples=set()):
        super().__init__(supervised_samples, unsupervised_samples)
        self.output_resolution = 129    # resolution of output image interpolation

        # Number of exclusions
        self.n_excl_dist = 'logn'                         # Number of exclusions distribution
        self.n_excl_dist_params = [7.8, .2]
        self.excl_margins = [.003, .003, .003, .003]     # closest distance of exclusion to boundary

        # Radii distribution
        self.r_dist = 'lognGP'
        self.r_dist_params = [-5.23, .3]
        self.r_GP_sigma = .4
        self.r_GP_length = .05

        # Center coordinate distribution
        self.coord_dist = 'GP'
        self.coord_dist_cov = 'squaredExponential'

        # Density of exclusions
        self.dens_length_scale = .08
        self.dens_sigm_scale = 1.2

        # Boundary conditions
        self.boundary_conditions = [.0, 1.0, 1.0, -0.0]

        # Mesh data
        self.vertex_coordinates = []
        self.cell_connectivity = []
        self.P = torch.zeros(self.n_supervised_samples, self.output_resolution**2)     # Pressure response at vertices
        self.V = []                                         # Velocity at vertices
        self.X = []
        self.microstructure_excl_radii = []
        self.microstructure_excl_centers = []

        self.imgX = []                                      # Microstructure pixel coordinates
        self.img_resolution = 256
        # Initialization -- dtype will change to self.dtype
        # Convention: supervised samples first, then unsupervised samples
        self.microstructure_image = torch.zeros(self.n_supervised_samples + self.n_unsupervised_samples,
                                                self.img_resolution**2, dtype=torch.bool)
        self.microstructure_image_inverted = torch.zeros(self.n_supervised_samples + self.n_unsupervised_samples,
                                                self.img_resolution ** 2, dtype=torch.bool)

    def set_path_name(self):
        assert len(self.path) == 0, 'Data path already set'
        self.path += '/home/constantin/'
        
        if socket.gethostname() == 'workstation1-room0436':
            # office workstation
            self.path += 'cluster/'
            
        self.path += 'python/data/stokesEquation/meshSize=' + str(self.resolution) + '/nonOverlappingDisks/margins=' + \
                     str(self.excl_margins[0]) + '_' + str(self.excl_margins[1]) + '_' + str(self.excl_margins[2]) +\
                     '_' + str(self.excl_margins[3]) + '/N~' + self.n_excl_dist + '/mu=' + \
                     str(self.n_excl_dist_params[0]) + '/sigma=' + str(self.n_excl_dist_params[1]) + \
                     '/x~' + self.coord_dist

        if self.coord_dist == 'GP':
            self.path += '/cov=' + self.coord_dist_cov + '/l=' + str(self.dens_length_scale) + '/sig_scale=' + \
                         str(self.dens_sigm_scale) + '/r~' + self.r_dist
        elif self.coord_dist == 'engineered' or self.coord_dist == 'tiles':
            self.path += '/r~' + self.r_dist
        else:
            self.path += '/mu=.5_.5' + '/cov=' + self.coord_dist_cov + '/r~' + self.r_dist

        if self.r_dist == 'lognGP':
            self.path += '/mu=' + str(self.r_dist_params[0]) + '/sigma=' + str(self.r_dist_params[1]) + '/sigmaGP_r=' +\
                            str(self.r_GP_sigma) + '/l=' + str(self.r_GP_length) + '/'
        else:
            self.path += '/mu=' + str(self.r_dist_params[0]) + '/sigma=' + str(self.r_dist_params[1]) + '/'

        self.solution_folder = self.path + 'p_bc=0.0/' + 'u_x=' + str(self.boundary_conditions[1]) + \
                          str(self.boundary_conditions[3]) + 'x[1]_u_y=' + str(self.boundary_conditions[2]) + \
                          str(self.boundary_conditions[3]) + 'x[0]'

    def read_data(self):
        if len(self.path) == 0 or len(self.solution_folder) == 0:
            # set path to file if not yet set
            self.set_path_name()

        for i, n in enumerate(self.supervised_samples):
            solutionFileName = self.solution_folder + '/solution' + str(n) + '.mat'
            file = sio.loadmat(solutionFileName)
            # self.P.append(file['p'])
            # self.X.append(file['x'])
            p = file['p']
            x = file['x']
            p_interp = self.interpolate_pressure(x, p, n)
            self.P[i, :] = torch.tensor(p_interp.flatten())
        
        for i, n in enumerate(self.supervised_samples | self.unsupervised_samples):
            try:
                self.microstructure_image[i] = torch.tensor(np.load((self.path + 'microstructure_image' +
                                                    str(n) + '_res=' + str(self.img_resolution)) + '.npy'),
                                                      dtype=torch.bool)
            except FileNotFoundError:
                # If not transformed to image yet --
                # should only happen for the first sample
                self.input2img(save=True)
                self.microstructure_image[i] = torch.tensor(np.load(self.path + 'microstructure_image' +
                                                      str(n) + '_res=' + str(self.img_resolution) + '.npy'),
                                                      dtype=torch.bool)

        self.microstructure_image = self.microstructure_image.type(self.dtype)
        # self.get_pixel_coordinates()

    def interpolate_pressure(self, x, p, n, shift=True):
        # shift shifts the data s.t. p(x = 0) = 0
        file_name = self.solution_folder + '/p_interp' + str(n) + '_res=' + str(self.output_resolution) + '.mat'
        file = sio.loadmat(file_name)
        p_interp = file['p_interp']
        if shift:
            p_interp -= p_interp[0]
        return p_interp

    def read_microstructure_information(self):
        if len(self.path) == 0:
            # set path to file if not yet set
            self.set_path_name()

        for n in (self.supervised_samples | self.unsupervised_samples):
            try:
                microstructureFileName = self.path + 'microstructureInformation' + str(n) + '.mat'
                microstructFile = sio.loadmat(microstructureFileName)
            except FileNotFoundError:
                microstructureFileName = self.path + 'microstructureInformation_nomesh' + str(n) + '.mat'
                microstructFile = sio.loadmat(microstructureFileName)
            self.microstructure_excl_radii.append(microstructFile['diskRadii'].flatten())
            self.microstructure_excl_centers.append(microstructFile['diskCenters'])

    def get_pixel_coordinates(self):
        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])
        self.imgX = torch.cat([xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)], 1)

    def input2img(self, save=False):
        if len(self.microstructure_excl_radii) == 0:
            self.read_microstructure_information()

        xx, yy = torch.meshgrid([torch.linspace(0, 1, self.img_resolution, dtype=self.dtype),
                                 torch.linspace(0, 1, self.img_resolution, dtype=self.dtype)])

        # loop over exclusions
        for i, n in enumerate(self.supervised_samples | self.unsupervised_samples):
            r2 = self.microstructure_excl_radii[i]**2.0

            for nEx in range(len(self.microstructure_excl_radii[i])):
                tmp = ((xx - self.microstructure_excl_centers[i][nEx, 0])**2.0 +
                       (yy - self.microstructure_excl_centers[i][nEx, 1])**2.0 <= r2[nEx])
                tmp = tmp.flatten()
                self.microstructure_image[i] = self.microstructure_image[i] | tmp
            # self.microstructure_image[-1] = self.microstructure_image[-1].type(self.dtype)
            if save:
                # save to pytorch tensor
                np.save(self.path + 'microstructure_image' + str(n) +
                           '_res=' + str(self.img_resolution), self.microstructure_image[i].detach().numpy())

        self.microstructure_image = self.microstructure_image.type(self.dtype)
        # CrossEntropyLoss wants dtype long == int64
        # self.microstructure_image = self.microstructure_image.type(torch.long)

    def reshape_microstructure_image(self):
        """
        Reshapes flattened input data to 2D image of img_resolution x img_resolution
        Should be a tensor ox shape (batchsize x channels x nPixelsH x nPixelsW)
        """
        tmp = torch.zeros(self.n_supervised_samples + self.n_unsupervised_samples, 1,
                          self.img_resolution, self.img_resolution, dtype=self.dtype)
        for i in range(self.n_supervised_samples + self.n_unsupervised_samples):
            tmp[i, 0, :, :] = torch.reshape(self.microstructure_image[i], (self.img_resolution, self.img_resolution))
        self.microstructure_image = tmp
        self.microstructure_image_inverted = 1.0 - tmp  # do once for performance


    def plot_microstruct(self, sampleNumber):
        if len(self.microstructure_image) == 0:
            self.input2img()
        plt.imshow(torch.reshape(self.microstructure_image[sampleNumber],
                                 (self.img_resolution, self.img_resolution)), cmap='binary')

