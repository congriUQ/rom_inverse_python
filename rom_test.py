import unittest
import ROM
from poisson_fem import PoissonFEM
import torch
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class IdentityInterpolationTest(unittest.TestCase):
    def setUp(self):
        self.lin_dim_rom = 8
        self.mesh = PoissonFEM.RectangularMesh(torch.ones(self.lin_dim_rom) / self.lin_dim_rom)

        def origin(x):
            return torch.abs(x[0]) < torch.finfo(torch.float32).eps and torch.abs(x[1]) < torch.finfo(torch.float32).eps

        def ess_boundary_fun(x):
            return 0.0

        self.mesh.set_essential_boundary(origin, ess_boundary_fun)

        def domain_boundary(x):
            # unit square
            return torch.abs(x[0]) < torch.finfo(torch.float32).eps or torch.abs(x[1]) < torch.finfo(
                torch.float32).eps or \
                   torch.abs(x[0]) > 1.0 - torch.finfo(torch.float32).eps or torch.abs(x[1]) > 1.0 - torch.finfo(
                torch.float32).eps

        self.mesh.set_natural_boundary(domain_boundary)

        def flux(x, bc_coeff=torch.tensor([1, 0, 0])):
            q = np.array([bc_coeff[0] + bc_coeff[2] * x[1], bc_coeff[1] + bc_coeff[2] * x[0]])
            return q

        # Specify right hand side and stiffness matrix
        self.rhs = PoissonFEM.RightHandSide(self.mesh)
        self.rhs.set_natural_rhs(self.mesh, flux)
        self.K = PoissonFEM.StiffnessMatrix(self.mesh)
        self.rhs.set_rhs_stencil(self.mesh, self.K)

        # define rom
        self.rom = ROM.ROM(self.mesh, self.K, self.rhs, (self.lin_dim_rom + 1) ** 2)
        # Interpolate on same square mesh.
        xx, yy = torch.meshgrid((torch.linspace(0, 1, self.lin_dim_rom + 1),
                                 torch.linspace(0, 1, self.lin_dim_rom + 1)))
        X = torch.cat((xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)), 1)
        self.rom.mesh.get_interpolation_matrix(X)

        self.rom_autograd = self.rom.get_autograd_fun()

    def test_identity_interpolation(self):
        lambdaf = torch.ones(self.lin_dim_rom**2)
        uf = self.rom_autograd(lambdaf)
        # print('uf == ', uf)
        # print('rom.full_solution.array == ', self.rom.full_solution.array)
        # print('rom.interpolated_solution.array == ', self.rom.interpolated_solution.array)
        # print('flip indices == ', self.mesh.glob_node_number_flip(torch.arange(self.mesh.n_vertices)))
        self.assertLess(np.linalg.norm(self.rom.full_solution.array - self.rom.interpolated_solution.array)/
                        np.linalg.norm(self.rom.full_solution.array), 1e-8)

if __name__ == '__main__':
    unittest.main()
