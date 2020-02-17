"""Module for Stokes and Darcy (Poisson) flow problems"""
import numpy as np

class FlowProblem:
    def __init__(self, a=np.array([.0, 1.0, .0, .0])):
        # Boundary conditions deduced from the pressure field p = a0 + ax*x = ay*y + axy*x*y,
        # u = - grad p = [-ax - axy*y, -ay - axy*x]
        self.a0 = a[0]
        self.ax = a[1]
        self.ay = a[2]
        self.axy = a[3]


class PoissonProblem(FlowProblem):
    def __init__(self, mesh, a):
        super().__init__(a)
        self.mesh = mesh
