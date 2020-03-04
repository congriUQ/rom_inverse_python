import torch
import matplotlib.pyplot as plt


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







