import torch
import sklearn.gaussian_process.kernels as kernels
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


class DiscretizedRandomField:
    """

    """
    def __init__(self, kernel=kernels.RBF(), discretization_vector=
                (torch.linspace(1/16, 15/16, 16), torch.linspace(1/16, 15/16, 16))):
        self.kernel = kernel
        self.gp_regressor = GaussianProcessRegressor(self.kernel)
        self.squashing_function = torch.exp

        assert isinstance(discretization_vector, (list, tuple)), "discretization_vector must be list/tuple of " \
                                                                 "1 or 2 discretization vectors"
        self.discretization_vector = discretization_vector
        if len(self.discretization_vector) < 2:
            # If discretization in y-direction is not given, use the same as in x-direction
            self.discretization_vector.append(self.discretization_vector)
        grid = torch.meshgrid(self.discretization_vector[0], self.discretization_vector[1])
        self.X = torch.cat((grid[0].flatten().unsqueeze(1), grid[1].flatten().unsqueeze(1)), 1)

    def sample(self, n_samples=1):
        """
        :param X: A dim x n_collocation_points numpy array of points where the GP is to be sampled
        :param n_samples: Number of Gaussian process realizations
        :return: The discretized realization
        """
        y_sample = torch.tensor(self.gp_regressor.sample_y(self.X.detach().numpy(), n_samples))
        return self.squashing_function(y_sample)

    def plot_realizations(self, n_samples=3):
        realization = self.sample(n_samples)
        fig = plt.figure(figsize=(30, 20))
        for r in range(realization.shape[1]):
            ax = plt.subplot(realization.shape[1], 1, r + 1)
            img = ax.imshow(realization[:, r].view(len(self.discretization_vector[0]),
                                             len(self.discretization_vector[1])))
            plt.colorbar(img)








