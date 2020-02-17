'''Poisson FEM base class'''
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sps
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from scipy.integrate import quad
import torch


class Mesh:
    def __init__(self):
        self.dtype = torch.float32

        self.vertices = []
        self.edges = []
        self.cells = []
        self.n_vertices = 0
        self._n_edges = 0
        self.n_cells = 0
        self.n_eq = None

        # Vector of zeros except of essential boundaries, where it holds the essential boundary value
        self.essential_solution_vector = None
        self.essential_solution_vector_torch = None
        # sparse matrix scattering solution vector from equation to vertex number
        self.scatter_matrix = None
        self.scatter_matrix_torch = None
        # interpolation matrix for reconstruction in p_cf
        self.interpolation_matrix = None
        self.interpolation_matrix_torch = None
        self.interpolation_matrix_sps = None

    def create_vertex(self, coordinates, globalVertexNumber=None, row_index=None, col_index=None):
        # Creates a vertex and appends it to vertex list
        vtx = Vertex(coordinates, globalVertexNumber, row_index, col_index)
        self.vertices.append(vtx)
        self.n_vertices += 1

    def create_edge(self, vtx0, vtx1):
        # Creates an edge between vertices specified by vertex indices and adds it to edge list
        edg = Edge(vtx0, vtx1)
        self.edges.append(edg)

        # Link edges to vertices
        vtx0.add_edge(edg)
        vtx1.add_edge(edg)

        self._n_edges += 1

    def create_cell(self, vertices, edges, number=None):
        # Creates a cell and appends it to cell list
        cll = Cell(vertices, edges, number)
        self.cells.append(cll)

        # Link cell to vertices
        for vtx in vertices:
            vtx.add_cell(cll)

        # Link cell to edges
        for edg in edges:
            edg.add_cell(cll)

        self.n_cells += 1

    def set_scatter_matrix(self):
        vtx_list = []
        eq_list = []
        for vtx in self.vertices:
            if not vtx.is_essential:
                vtx_list.append(vtx.global_number)
                eq_list.append(vtx.equation_number)
        indices = torch.LongTensor([vtx_list, eq_list])

        self.scatter_matrix_torch = torch.sparse.FloatTensor(indices, torch.ones(self.n_eq),
                                                       torch.Size([self.n_vertices, self.n_eq]))

        # dense performs better for the time being -- change back here if that changes
        self.scatter_matrix_torch = self.scatter_matrix_torch.to_dense()

        # Best performance has PETSc
        tmp = sps.csr_matrix(self.scatter_matrix_torch)
        self.scatter_matrix = PETSc.Mat().createAIJ(size=tmp.shape, csr=(tmp.indptr, tmp.indices, tmp.data))

    def set_essential_boundary(self, location_indicator_fun, value_fun):
        # location_indicator_fun is the indicator function to the essential boundary
        equation_number = 0
        for vtx in self.vertices:
            if location_indicator_fun(vtx.coordinates):
                vtx.is_essential = True
                vtx.boundary_value = value_fun(vtx.coordinates)
                self.essential_solution_vector[vtx.global_number] = vtx.boundary_value
                for cll in vtx.cells:
                    cll.contains_essential_vertex = True
            else:
                vtx.equation_number = equation_number
                equation_number += 1
        self.essential_solution_vector_torch = \
            torch.tensor(self.essential_solution_vector.array, dtype=self.dtype).unsqueeze(1)
        self.n_eq = equation_number

        self.set_scatter_matrix()

    def set_natural_boundary(self, location_indicator_fun):
        # location_indicator_fun is the indicator function to the natural boundary
        # location_indicator_fun should include pointwise essential boundaries, i.e., for essential boundary at
        # x = 0, y = 0, location_indicator_fun should indicate the whole domain boundary
        for edg in self.edges:
            if location_indicator_fun(edg.vertices[0].coordinates) \
                    and location_indicator_fun(edg.vertices[1].coordinates):
                # Both edge vertices are on natural boundary, i.e., edge is natural boundary
                edg.is_natural = True

    def get_interpolation_matrix(self, x):
        # returns the shape function interpolation matrix on x
        # x is a np.array of shape N x 2

        # If points are 'inside' of multiple cells (on edges/vertices),
        # the shape function values need to be divided accordingly
        isin_sum = torch.zeros(x.shape[0])
        for cll in self.cells:
            isin_sum += cll.is_inside(x)

        # if W is a free parameter, set requires_grad=True
        W = torch.zeros((x.shape[0], self.n_vertices))
        for cll in self.cells:
            # arrays
            shape_fun_values, glob_vertex_numbers = cll.element_shape_function_values(x)
            for loc_vertex in range(4):
                W[:, glob_vertex_numbers[loc_vertex]] += shape_fun_values[loc_vertex]/isin_sum

        # Convert to PETSc sparse matrix for efficiency?
        self.interpolation_matrix_torch = W
        W = sps.csr_matrix(W)
        self.interpolation_matrix = PETSc.Mat().createAIJ(size=W.shape, csr=(W.indptr, W.indices, W.data))

        self.interpolation_matrix_sps = W

    def plot(self):
        for vtx in self.vertices:
            if vtx is not None:
                vtx.plot()

        n = 0
        for edg in self.edges:
            if edg is not None:
                edg.plot(n)
                n += 1

        # plot cell number into cell
        for cll in self.cells:
            if cll is not None:
                plt.text(cll.centroid[0], cll.centroid[1], str(cll.number))

        plt.xlabel('x')
        plt.ylabel('y')


class RectangularMesh(Mesh):
    # Assuming a unit square domain
    def __init__(self, grid_x=np.ones(4)/4, grid_y=None):
        # Grid vectors grid_x, grid_y give the edge lengths

        super().__init__()
        if grid_y is None:
            # assume square mesh
            grid_y = grid_x

        # Create vertices
        x_coord = np.concatenate((np.array([.0]), np.cumsum(grid_x)))
        y_coord = np.concatenate((np.array([.0]), np.cumsum(grid_y)))
        n = 0
        for row_index, y in enumerate(y_coord):
            for col_index, x in enumerate(x_coord):
                self.create_vertex(np.array([x, y]), globalVertexNumber=n, row_index=row_index, col_index=col_index)
                n += 1

        self.essential_solution_vector = PETSc.Vec().createSeq(self.n_vertices)

        # Create edges
        nx = len(grid_x) + 1
        ny = len(grid_y) + 1
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x + y*nx].coordinates[0] > 0:
                    self.create_edge(self.vertices[x + y*nx - 1], self.vertices[x + y*nx])
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x + y*nx].coordinates[1] > 0:
                    self.create_edge(self.vertices[x + (y - 1)*nx], self.vertices[x + y*nx])

        # Create cells
        nx -= 1
        ny -= 1
        n = 0   # cell index
        for y in range(ny):
            for x in range(nx):
                vtx = [self.vertices[x + y*(nx + 1)], self.vertices[x + y*(nx + 1) + 1],
                       self.vertices[x + (y + 1)*(nx + 1) + 1], self.vertices[x + (y + 1)*(nx + 1)]]
                edg = [self.edges[n], self.edges[nx*(ny + 1) + n + y + 1], self.edges[n + nx],
                       self.edges[nx*(ny + 1) + n + y]]
                self.create_cell(vtx, edg, number=n)
                n += 1

        # minor stuff
        self.n_el_x = len(grid_x)
        self.n_el_y = len(grid_y)

    def state_dict(self):
        return {'dtype': self.dtype,
                'vertices': self.vertices,
                'edges': self.edges,
                'cells': self.cells,
                'n_vertices': self.n_vertices,
                'n_edges': self._n_edges,
                'n_cells': self.n_cells,
                'n_eq': self.n_eq,
                'essential_solution_vector': self.essential_solution_vector.array,
                'scatter_matrix': self.scatter_matrix_torch,
                'interpolation_matrix': self.interpolation_matrix_sps,
                'n_el_x': self.n_el_x,
                'n_el_y': self.n_el_y}

    def save(self, path='./mesh.p'):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path='./mesh.p'):
        # loads mesh from pickle file
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        # loads mesh from state_dict
        self.dtype = state_dict['dtype']
        self.vertices = state_dict['vertices']
        self.edges = state_dict['edges']
        self.cells = state_dict['cells']
        self.n_vertices = state_dict['n_vertices']
        self._n_edges = state_dict['n_edges']
        self.n_cells = state_dict['n_cells']
        self.n_eq = state_dict['n_eq']
        self.essential_solution_vector = PETSc.Vec().createSeq(self.n_vertices)
        self.essential_solution_vector.array = state_dict['essential_solution_vector']
        self.essential_solution_vector_torch = torch.tensor(self.essential_solution_vector.array, dtype=self.dtype)
        self.scatter_matrix_torch = state_dict['scatter_matrix']
        tmp = sps.csr_matrix(self.scatter_matrix_torch)
        self.scatter_matrix = PETSc.Mat().createAIJ(size=tmp.shape, csr=(tmp.indptr, tmp.indices, tmp.data))
        self.interpolation_matrix_sps = state_dict['interpolation_matrix']
        W_tmp = self.interpolation_matrix_sps
        self.interpolation_matrix = \
            PETSc.Mat().createAIJ(size=W_tmp.shape, csr=(W_tmp.indptr, W_tmp.indices, W_tmp.data))
        self.n_el_x = state_dict['n_el_x']
        self.n_el_y = state_dict['n_el_y']


class Cell:
    def __init__(self, vertices, edges, number=None):
        # Vertices and edges must be sorted according to local vertex/edge number!
        self.vertices = vertices
        self.edges = edges
        self.number = number
        self.centroid = []
        self.compute_centroid()
        self.surface = self.edges[0].length*self.edges[1].length
        self.contains_essential_vertex = False

    def compute_centroid(self):
        # Compute cell centroid
        self.centroid = np.zeros(2)
        for vtx in self.vertices:
            self.centroid += vtx.coordinates
        self.centroid /= len(self.vertices)

    def delete_edges(self, indices):
        # Deletes edges according to indices by setting them to None
        for i in indices:
            self.edges[i] = None

    def is_inside(self, x):
        # Checks if point x is inside of cell
        return np.logical_and(np.logical_and(np.logical_and(
                np.less(self.vertices[0].coordinates[0] - np.finfo(float).eps, x[:, 0]),
                np.less_equal(x[:, 0], self.vertices[2].coordinates[0] + np.finfo(float).eps)),
                np.less(self.vertices[0].coordinates[1] - np.finfo(float).eps, x[:, 1])),
                np.less_equal(x[:, 1], self.vertices[2].coordinates[1] + np.finfo(float).eps))

    def element_shape_function_values(self, x):
        # Returns a list of the values [N_0^(c)(x), N_1^(c)(x), N_2^(c)(x), N_3^(c)(x)] of the element shape
        # functions of x, and a list [global_number0, ...,  global_number3] of global vertex numbers
        shape_fun_values = []
        glob_vertex_numbers = []
        isin = self.is_inside(x)
        shape_fun_values.append(isin*(x[:, 0] - self.vertices[1].coordinates[0]) *
                                        (x[:, 1] - self.vertices[3].coordinates[1]) / self.surface)
        shape_fun_values.append(isin*(-(x[:, 0] - self.vertices[0].coordinates[0]) *
                                        (x[:, 1] - self.vertices[3].coordinates[1]) / self.surface))
        shape_fun_values.append(isin*(x[:, 0] - self.vertices[0].coordinates[0]) *
                                        (x[:, 1] - self.vertices[0].coordinates[1]) / self.surface)
        shape_fun_values.append(isin*(-(x[:, 0] - self.vertices[1].coordinates[0]) *
                                        (x[:, 1] - self.vertices[0].coordinates[1]) / self.surface))
        glob_vertex_numbers = [vtx.global_number for vtx in self.vertices]

        return shape_fun_values, glob_vertex_numbers


class Edge:
    def __init__(self, vtx0, vtx1):
        self.vertices = [vtx0, vtx1]
        self.cells = []
        self.length = np.linalg.norm(vtx0.coordinates - vtx1.coordinates)
        self.is_natural = False       # True if edge is on natural boundary

    def add_cell(self, cell):
        self.cells.append(cell)

    def plot(self, n):
        plt.plot([self.vertices[0].coordinates[0], self.vertices[1].coordinates[0]],
                 [self.vertices[0].coordinates[1], self.vertices[1].coordinates[1]], linewidth=.5, color='r')
        plt.text(.5*(self.vertices[0].coordinates[0] + self.vertices[1].coordinates[0]),
                 .5*(self.vertices[0].coordinates[1] + self.vertices[1].coordinates[1]), str(n), color='r')


class Vertex:
    def __init__(self, coordinates=np.zeros((2, 1)), global_number=None, row_index=None, col_index=None):
        self.coordinates = coordinates
        self.cells = []
        self.edges = []
        self.is_essential = False    # False for natural, True for essential vertex
        self.boundary_value = None    # Value for essential boundary
        self.equation_number = None   # Equation number of dof belonging to vertex
        self.global_number = global_number
        self.row_index = row_index
        self.col_index = col_index

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_edge(self, edge):
        self.edges.append(edge)

    def plot(self):
        p = plt.plot(self.coordinates[0], self.coordinates[1], 'bx', linewidth=2.0, markersize=8.0)
        plt.text(self.coordinates[0], self.coordinates[1], str(self.global_number), color='b')


class FunctionSpace:
    # Only bilinear is implemented!
    def __init__(self, mesh, typ='bilinear'):
        self.shape_function_gradients = None
        self.get_shape_function_gradient_matrix(mesh)

    def get_shape_function_gradient_matrix(self, mesh):
        # Computes shape function gradient matrices B, see Fish & Belytshko

        # Gauss quadrature points
        xi0 = -1.0 / np.sqrt(3)
        xi1 = 1.0 / np.sqrt(3)

        self.shape_function_gradients = np.empty((8, 4, mesh.n_cells))
        for e in range(mesh.n_cells):
            # short hand notation
            x0 = mesh.cells[e].vertices[0].coordinates[0]
            x1 = mesh.cells[e].vertices[1].coordinates[0]
            y0 = mesh.cells[e].vertices[0].coordinates[1]
            y3 = mesh.cells[e].vertices[3].coordinates[1]

            # coordinate transformation
            xI = .5 * (x0 + x1) + .5 * xi0 * (x1 - x0)
            xII = .5 * (x0 + x1) + .5 * xi1 * (x1 - x0)
            yI = .5 * (y0 + y3) + .5 * xi0 * (y3 - y0)
            yII = .5 * (y0 + y3) + .5 * xi1 * (y3 - y0)

            # B matrices for bilinear shape functions
            B0 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xI - x1, x0 - xI, xI - x0, x1 - xI]])
            B1 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B2 = np.array([[yI - y3, y3 - yI, yI - y0, y0 - yI], [xII - x1, x0 - xII, xII - x0, x1 - xII]])
            B3 = np.array([[yII - y3, y3 - yII, yII - y0, y0 - yII], [xI - x1, x0 - xI, xI - x0, x1 - xI]])

            # Note:in Gauss quadrature, the differential transforms as dx = (l_x/2) d xi. Hence we take the
            # additional factor of sqrt(A)/2 onto B
            self.shape_function_gradients[:, :, e] = (1 / (2 * np.sqrt(mesh.cells[e].surface))) * np.concatenate(
                (B0, B1, B2, B3))

    def state_dict(self):
        return {'shape_function_gradients': self.shape_function_gradients}


class StiffnessMatrix:
    def __init__(self, mesh):
        self.mesh = mesh
        self.funSpace = FunctionSpace(mesh)

        # Set up solver
        self.solver = PETSc.KSP().create()
        self.solver.setType('preonly')
        precond = self.solver.getPC()
        precond.setType('cholesky')
        self.solver.setFromOptions()  # ???

        self.loc_stiff_grad = None
        # note the permutation; this is for fast computation of gradients via adjoints
        self.glob_stiff_grad = torch.empty((mesh.n_eq, mesh.n_cells, mesh.n_eq))
        self.glob_stiff_stencil = None
        self.glob_stiff_stencil_scipy = None

        # Stiffness sparsity pattern
        self.nnz = None             # number of nonzero components
        self.vec_nonzero = None     # nonzero component indices of flattened stiffness matrix
        self.indptr = None          # csr indices
        self.indices = None

        # Pre-computations
        self.compute_glob_stiff_stencil()
        self.find_sparsity_pattern()

        # Pre-allocations
        self.matrix = PETSc.Mat().createAIJ(size=(mesh.n_eq, mesh.n_eq), nnz=self.nnz)
        self.matrix_torch = None
        self.cholesky_L = None
        # self.conductivity = PETSc.Vec().createSeq(mesh.n_cells)     # permeability/diffusivity vector
        self.assembly_vector = PETSc.Vec().createSeq(mesh.n_eq**2)      # For quick assembly with matrix vector product

    def compute_equation_indices(self):
        # Compute equation indices for direct assembly of stiffness matrix
        equationIndices0 = np.array([], dtype=np.uint32)
        equationIndices1 = np.array([], dtype=np.uint32)
        locIndices0 = np.array([], dtype=np.uint32)
        locIndices1 = np.array([], dtype=np.uint32)
        cllIndex = np.array([], dtype=np.uint32)

        for cll in self.mesh.cells:
            equations = np.array([], dtype=np.uint32)
            eqVertices = np.array([], dtype=np.uint32)
            for v, vtx in enumerate(cll.vertices):
                if vtx.equation_number is not None:
                    equations = np.append(equations, np.array([vtx.equation_number], dtype=np.uint32))
                    eqVertices = np.append(eqVertices, np.array([v], dtype=np.uint32))

            eq0, eq1 = np.meshgrid(equations, equations)
            vtx0, vtx1 = np.meshgrid(eqVertices, eqVertices)
            equationIndices0 = np.append(equationIndices0, eq0.flatten())
            equationIndices1 = np.append(equationIndices1, eq1.flatten())
            locIndices0 = np.append(locIndices0, vtx0.flatten())
            locIndices1 = np.append(locIndices1, vtx1.flatten())
            cllIndex = np.append(cllIndex, cll.number*np.ones_like(vtx0.flatten()))
        k_index = np.ravel_multi_index((locIndices1, locIndices0, cllIndex), (4, 4, self.mesh.n_cells), order='F')
        return [equationIndices0, equationIndices1], k_index

    def compute_loc_stiff_grad(self):
        # Compute local stiffness matrix gradients w.r.t. diffusivities
        if self.funSpace.shape_function_gradients is None:
            self.funSpace.get_shape_function_gradient_matrix()

        self.loc_stiff_grad = self.mesh.n_cells * [np.empty((4, 4))]
        for e in range(self.mesh.n_cells):
            self.loc_stiff_grad[e] = np.transpose(self.funSpace.shape_function_gradients[:, :, e])\
                                     @ self.funSpace.shape_function_gradients[:, :, e]

    def compute_glob_stiff_stencil(self):
        # Compute stiffness stencil matrices K_e, such that K can be assembled via K = sum_e (lambda_e*K_e)
        # This can be done much more efficiently, but is precomputed and therefore not bottleneck.
        if self.loc_stiff_grad is None:
            self.compute_loc_stiff_grad()

        equation_indices, k_index = self.compute_equation_indices()

        glob_stiff_stencil = np.empty((self.mesh.n_eq**2, self.mesh.n_cells))

        for e, cll in enumerate(self.mesh.cells):
            grad_loc_k = np.zeros((4, 4, self.mesh.n_cells))
            grad_loc_k[:, :, e] = self.loc_stiff_grad[e]
            grad_loc_k = grad_loc_k.flatten(order='F')
            Ke = sps.csr_matrix((grad_loc_k[k_index], (equation_indices[0], equation_indices[1])))
            Ke_dense = sps.csr_matrix.todense(Ke)
            # Ke = sps.csr_matrix(Ke_dense)
            self.glob_stiff_grad[:, e, :] = torch.tensor(Ke_dense)
            glob_stiff_stencil[:, e] = Ke_dense.flatten(order='F')

        self.glob_stiff_stencil_scipy = sps.csr_matrix(glob_stiff_stencil)
        glob_stiff_stencil = PETSc.Mat().createAIJ(
            size=glob_stiff_stencil.shape, csr=(self.glob_stiff_stencil_scipy.indptr,
                                                self.glob_stiff_stencil_scipy.indices,
                                                self.glob_stiff_stencil_scipy.data))
        glob_stiff_stencil.assemblyBegin()
        glob_stiff_stencil.assemblyEnd()

        self.glob_stiff_stencil = glob_stiff_stencil


    def find_sparsity_pattern(self):
        # Computes sparsity pattern of stiffness matrix/stiffness matrix vector for fast matrix assembly
        test_vec = PETSc.Vec().createSeq(self.mesh.n_cells)
        test_vec.setValues(range(self.mesh.n_cells), np.ones(self.mesh.n_cells))
        Kvec = PETSc.Vec().createSeq(self.mesh.n_eq**2)

        self.glob_stiff_stencil.mult(test_vec, Kvec)
        self.vec_nonzero = np.nonzero(Kvec.array)
        self.nnz = len(self.vec_nonzero[0])     # important for memory allocation

        Ktmp = sps.csr_matrix(np.reshape(Kvec.array, (self.mesh.n_eq, self.mesh.n_eq), order='F'))
        self.indptr = Ktmp.indptr.copy()     # csr-like indices
        self.indices = Ktmp.indices.copy()

    def assemble(self, x):
        # x is numpy vector of permeability/conductivity
        self.glob_stiff_stencil.mult(x, self.assembly_vector)
        self.matrix.setValuesCSR(self.indptr, self.indices, self.assembly_vector.getValues(self.vec_nonzero))
        self.matrix.assemblyBegin()
        self.matrix.assemblyEnd()

        # Everytime the stiffness matrix changes, the solver operators need to be reset
        self.solver.setOperators(self.matrix)

    def assemble_torch(self, x):
        """
        For batched computation
        x is a torch tensor of permeabilities of shape batch_size x n_cells
        """
        self.matrix_torch = torch.tensor(np.moveaxis((self.glob_stiff_stencil_scipy @ x.squeeze().T.detach().numpy())
                                                     .reshape(self.mesh.n_eq, self.mesh.n_eq, x.shape[0]), -1, 0),
                                         dtype=self.mesh.dtype)


class RightHandSide:
    # This is the finite element force vector
    def __init__(self, mesh):
        self.vector = PETSc.Vec().createSeq(mesh.n_eq)    # Force vector
        self.vector_torch = None
        self.flux_boundary_condition = None
        self.source_field = None
        self.natural_rhs = PETSc.Vec().createSeq(mesh.n_eq)
        self.natural_rhs_torch = None
        self.cells_with_essential_boundary = []    # precompute for performance
        self.find_cells_with_essential_boundary(mesh)
        # Use nnz = 0, PETSc will allocate  additional storage by itself
        self.rhs_stencil = PETSc.Mat().createAIJ(size=(mesh.n_eq, mesh.n_cells), nnz=0)
        self.rhs_stencil.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.rhs_stencil_torch = torch.zeros((mesh.n_eq, mesh.n_cells), dtype=mesh.dtype)

    def set_flux_boundary_condition(self, mesh, flux):
        # Contribution due to flux boundary conditions

        self.flux_boundary_condition = np.zeros((4, mesh.n_cells))
        for cll in mesh.cells:
            for edg in cll.edges:
                if edg.is_natural:
                    # Edge is a natural boundary
                    # This is only valid for the unit square domain!
                    # Find the edge outward normal vectors for generalization

                    # Short hand notation
                    ll = cll.vertices[0].coordinates    # lower left
                    ur = cll.vertices[2].coordinates    # upper right
                    if edg.vertices[0].coordinates[1] < np.finfo(float).eps and \
                       edg.vertices[1].coordinates[1] < np.finfo(float).eps:
                        # lower boundary
                        for i in range(4):
                            def fun(x):
                                q = flux(np.array([x, 0.0]))
                                q = - q[1]      # scalar product with boundary
                                if i == 0:
                                    N = (x - ur[0]) * (-ur[1])
                                elif i == 1:
                                    N = -(x - ll[0])*(- ur[1])
                                elif i == 2:
                                    N = (x - ll[0]) * (- ll[1])
                                elif i == 3:
                                    N = -(x - ur[0])*(- ll[1])
                                N /= cll.surface
                                return q*N

                            Intgrl = quad(fun, ll[0], ur[0])
                            self.flux_boundary_condition[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[0] > 1.0 - np.finfo(float).eps and \
                         edg.vertices[1].coordinates[0] > 1.0 - np.finfo(float).eps:
                        # right boundary
                        for i in range(4):
                            def fun(y):
                                q = flux(np.array([1.0, y]))
                                q = q[0]  # scalar product with boundary
                                if i == 0:
                                    N = (1.0 - ur[0]) * (y - ur[1])
                                elif i == 1:
                                    N = - (1.0 - ll[0]) * (y - ur[1])
                                elif i == 2:
                                    N = (1.0 - ll[0]) * (y - ll[1])
                                elif i == 3:
                                    N = - (1.0 - ur[0]) * (y - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[1], ur[1])
                            self.flux_boundary_condition[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[1] > 1.0 - np.finfo(float).eps and \
                         edg.vertices[1].coordinates[1] > 1.0 - np.finfo(float).eps:
                        # upper boundary
                        for i in range(4):
                            def fun(x):
                                q = flux(np.array([x, 1.0]))
                                q = q[1]  # scalar product with boundary
                                if i == 0:
                                    N = (x - ur[0]) * (1.0 - ur[1])
                                elif i == 1:
                                    N = - (x - ll[0]) * (1.0 - ur[1])
                                elif i == 2:
                                    N = (x - ll[0]) * (1.0 - ll[1])
                                elif i == 3:
                                    N = - (x - ur[0]) * (1.0 - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[0], ur[0])
                            self.flux_boundary_condition[i, cll.number] += Intgrl[0]

                    elif edg.vertices[0].coordinates[0] < np.finfo(float).eps and \
                         edg.vertices[1].coordinates[0] < np.finfo(float).eps:
                        # left boundary
                        for i in range(4):
                            def fun(y):
                                q = flux(np.array([0.0, y]))
                                q = -q[0]  # scalar product with boundary
                                if i == 0:
                                    N = (- ur[0]) * (y - ur[1])
                                elif i == 1:
                                    N = ll[0] * (y - ur[1])
                                elif i == 2:
                                    N = ( - ll[0]) * (y - ll[1])
                                elif i == 3:
                                    N = ur[0] * (y - ll[1])
                                N /= cll.surface
                                return q * N

                            Intgrl = quad(fun, ll[1], ur[1])
                            self.flux_boundary_condition[i, cll.number] += Intgrl[0]

    def set_source_field(self, mesh, source_field):
        # Local force vector elements due to source field

        # Gauss points
        xi0 = -1.0/np.sqrt(3)
        xi1 = 1.0/np.sqrt(3)
        eta0 = -1.0/np.sqrt(3)
        eta1 = 1.0/np.sqrt(3)

        self.source_field = np.zeros((4, mesh.n_cells))

        for c, cll in enumerate(mesh.cells):
            # Short hand notation
            x0 = cll.vertices[0].coordinates[0]
            x1 = cll.vertices[2].coordinates[0]
            y0 = cll.vertices[0].coordinates[1]
            y1 = cll.vertices[2].coordinates[1]

            # Coordinate transformation
            xI = .5*(x0 + x1) + .5*xi0*(x1 - x0)
            xII = .5*(x0 + x1) + .5*xi1*(x1 - x0)
            yI = .5*(y0 + y1) + .5*eta0*(y1 - y0)
            yII = .5*(y0 + y1) + .5*eta1*(y1 - y0)

            source = source_field(cll.centroid)
            self.source_field[0, c] = source*(1.0/cll.surface)*((xI - x1)*(yI - y1) + (xII - x1)*(yII - y1) +
                                                               (xI - x1)*(yII - y1) + (xII - x1)*(yI - y1))
            self.source_field[1, c] = - source*(1.0/cll.surface)*((xI - x0)*(yI - y1) + (xII - x0) * (yII - y1) +
                                                                 (xI - x0)*(yII - y1) + (xII - x0)*(yI - y1))
            self.source_field[2, c] = source*(1.0/cll.surface)*((xI - x0)*(yI - y0) + (xII - x0)*(yII - y0) +
                                                               (xI - x0)*(yII - y0) + (xII - x0) * (yI - y0))
            self.source_field[3, c] = - source*(1.0/cll.surface)*((xI - x1)*(yI - y0) + (xII - x1)*(yII - y0) +
                                                                 (xI - x1)*(yII - y0) + (xII - x1)*(yI - y0))

    def set_natural_rhs(self, mesh, flux):
        # Sets the part of the RHS due to natural BC's and source field
        if self.flux_boundary_condition is None:
            self.set_flux_boundary_condition(mesh, flux)

        natural_rhs = np.zeros(mesh.n_eq)
        for e, cll in enumerate(mesh.cells):
            for v, vtx in enumerate(cll.vertices):
                if vtx.equation_number is not None:
                    natural_rhs[vtx.equation_number] += self.flux_boundary_condition[v, e]
                    if self.source_field is not None:
                        natural_rhs[vtx.equation_number] += self.source_field[v, e]

        self.natural_rhs.setValues(range(mesh.n_eq), natural_rhs)
        self.natural_rhs_torch = torch.tensor(self.natural_rhs.getArray(), dtype=mesh.dtype).unsqueeze(1)

    def find_cells_with_essential_boundary(self, mesh):
        for cll in mesh.cells:
            if cll.contains_essential_vertex:
                self.cells_with_essential_boundary.append(cll.number)

    def set_rhs_stencil(self, mesh, stiffness_matrix):
        # rhs_stencil_np = np.zeros((mesh.n_eq, mesh.n_cells))
        for c in self.cells_with_essential_boundary:
            essential_boundary_values = np.zeros(4)
            for v, vtx in enumerate(mesh.cells[c].vertices):
                if vtx.is_essential:
                    essential_boundary_values[v] = vtx.boundary_value

            loc_ess_bc = stiffness_matrix.loc_stiff_grad[c] @ essential_boundary_values
            for v, vtx in enumerate(mesh.cells[c].vertices):
                if vtx.equation_number is not None:
                    self.rhs_stencil_torch[vtx.equation_number, c] -= loc_ess_bc[v]

        # Assemble PETSc matrix from numpy
        for c in self.cells_with_essential_boundary:
            for vtx in mesh.cells[c].vertices:
                if vtx.equation_number is not None:
                    self.rhs_stencil.setValue(vtx.equation_number, c, self.rhs_stencil_torch[vtx.equation_number, c])
        self.rhs_stencil.assemblyBegin()
        self.rhs_stencil.assemblyEnd()

    def expand_rhs_stencil_torch(self, batch_size):
        """
        Expands the rhs stencil to batch size
        :param batch_size:
        :return:
        """
        self.rhs_stencil_torch = self.rhs_stencil_torch.unsqueeze(0).expand(batch_size, -1, -1)

    def assemble(self, x):
        # x is a PETSc vector of conductivity/permeability
        self.rhs_stencil.multAdd(x, self.natural_rhs, self.vector)

    def assemble_torch(self, x):
        """
        x is a batch_size x n_cells x 1 torch tensor of cell permeabilities
        :param x:
        :return:
        """
        self.vector_torch = self.natural_rhs_torch + torch.bmm(self.rhs_stencil_torch, x)

    def state_dict(self):
        return {'vector': self.vector.array,
                'natural_rhs': self.natural_rhs.array,
                'cells_with_essential_boundary': self.cells_with_essential_boundary}

