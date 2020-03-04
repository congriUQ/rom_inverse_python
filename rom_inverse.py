import torch
import sklearn.gaussian_process.kernels as kernels
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import os
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pyro.contrib.gp as gp
smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.2.1')
pyro.enable_validation(True)       # can help with debugging


class DiscretizedRandomField:
    """

    """
    def __init__(self, resolution, kernel=gp.kernels.RBF(input_dim=2), nugget=1e-5):
        self.kernel = kernel
        self.nugget = nugget # to make the covariance matrix positive definite
        # self.gp_regressor = GaussianProcessRegressor(self.kernel)
        self.squashing_function = torch.exp

        if len(resolution) < 2:
            self.discretization_vector = torch.linspace(.5/resolution[0], (resolution[0] - .5)/resolution[0],
                                                        resolution[0])
            self.discretization_vector = (self.discretization_vector, self.discretization_vector)
        else:
            self.discretization_vector = (torch.linspace(.5/resolution[0], (resolution[0] - .5)/resolution[0],
                                                         resolution[0]),
                                     torch.linspace(.5/resolution[1], (resolution[1] - .5)/resolution[1], resolution[1]))

        grid = torch.meshgrid(self.discretization_vector[0], self.discretization_vector[1])
        self.X = torch.cat((grid[0].flatten().unsqueeze(1), grid[1].flatten().unsqueeze(1)), 1)
        self.log_permeability_cov = None
        self.log_permeability_scale_tril = None

    def set_covariance_matrix(self):
        # this can be memory intensive for large systems!
        if self.log_permeability_cov is None:
            self.log_permeability_cov = self.kernel.forward(self.X) + self.nugget*torch.eye(self.X.shape[0])
            self.log_permeability_scale_tril = torch.cholesky(self.log_permeability_cov)
        # return self.log_permeability_cov

    def sample_log_permeability(self, n_samples=1):
        """
        :param X: A dim x n_collocation_points numpy array of points where the GP is to be sampled
        :param n_samples: Number of Gaussian process realizations
        :return: The discretized realization
        """
        self.set_covariance_matrix()  # can be memory consuming for large systems
        samples = dist.MultivariateNormal(torch.zeros(self.X.shape[0]), scale_tril=self.log_permeability_scale_tril).\
            sample(sample_shape=(n_samples,))
        # sample = torch.tensor(self.gp_regressor.sample_y(self.X.detach().numpy(), n_samples))
        return samples

    def sample_permeability(self, n_samples=1):
        """
        :param X: A dim x n_collocation_points numpy array of points where the GP is to be sampled
        :param n_samples: Number of Gaussian process realizations
        :return: The discretized realization
        """
        samples = self.sample_log_permeability(n_samples)
        return self.squashing_function(samples)

    def plot_realizations(self, n_samples=3):
        realization = self.sample_permeability(n_samples)
        fig = plt.figure(figsize=(30, 20))
        for r in range(realization.shape[0]):
            ax = plt.subplot(realization.shape[0], 1, r + 1)
            img = ax.imshow(realization[r, :].view(len(self.discretization_vector[0]),
                                             len(self.discretization_vector[1])))
            plt.colorbar(img)








