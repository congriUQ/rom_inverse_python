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
    def __init__(self, kernel=gp.kernels.RBF(input_dim=2),
                 discretization_vector=(torch.linspace(.5/16, 15.5/16, 16), torch.linspace(.5/16, 15.5/16, 16)),
                 nugget=1e-5):
        self.kernel = kernel
        self.nugget = nugget # to make the covariance matrix positive definite
        # self.gp_regressor = GaussianProcessRegressor(self.kernel)
        self.squashing_function = torch.exp

        assert isinstance(discretization_vector, (list, tuple)), "discretization_vector must be list/tuple of " \
                                                                 "1 or 2 discretization vectors"
        self.discretization_vector = discretization_vector
        if len(self.discretization_vector) < 2:
            # If discretization in y-direction is not given, use the same as in x-direction
            self.discretization_vector.append(self.discretization_vector)
        grid = torch.meshgrid(self.discretization_vector[0], self.discretization_vector[1])
        self.X = torch.cat((grid[0].flatten().unsqueeze(1), grid[1].flatten().unsqueeze(1)), 1)
        self.log_permeability_cov = None

    def get_covariance_matrix(self):
        # this can be memory intensive for large systems!
        if self.log_permeability_cov is None:
            self.log_permeability_cov = self.kernel.forward(self.X) + self.nugget*torch.eye(self.X.shape[0])
        return self.log_permeability_cov

    def sample_log_permeability(self, n_samples=1):
        """
        :param X: A dim x n_collocation_points numpy array of points where the GP is to be sampled
        :param n_samples: Number of Gaussian process realizations
        :return: The discretized realization
        """
        cov = self.get_covariance_matrix()  # can be memory consuming for large systems
        samples = dist.MultivariateNormal(torch.zeros(self.X.shape[0]), covariance_matrix=cov). \
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








