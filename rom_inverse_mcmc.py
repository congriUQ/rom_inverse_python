import rom_inverse as ri
import torch
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from poisson_fem import PoissonFEM
import ROM
import numpy as np
import scipy as sp
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import os
import pyro.contrib.gp as gp
from datetime import datetime
smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.2.1')
pyro.enable_validation(True)       # can help with debugging

print('inverse temperature beta == ', beta := 20)
print('fom resolution == ', lin_dim_fom := 16, ' x ', lin_dim_fom)               # Linear number of rom elements
a = torch.tensor([1, 1, 0])               # Boundary condition function coefficients
permeability_length_scale = torch.tensor([.3, .1])
permeability_variance = torch.tensor(2.)
n_chains = 4
n_samples = 500
n_warmup = 100


kernel = gp.kernels.RBF(input_dim=2, variance=permeability_variance, lengthscale=permeability_length_scale)
permeability_random_field = ri.DiscretizedRandomField(kernel=kernel,
                                discretization_vector=(torch.linspace(1/(2*lin_dim_fom),
                                (lin_dim_fom - .5)/lin_dim_fom, lin_dim_fom),
                                torch.linspace(1/(2*lin_dim_fom), (lin_dim_fom - .5)/lin_dim_fom, lin_dim_fom)),
                                                      nugget=1e-3)
mesh = PoissonFEM.RectangularMesh(torch.ones(lin_dim_fom)/lin_dim_fom)


def origin(x):
    return torch.abs(x[0]) < torch.finfo(torch.float32).eps and torch.abs(x[1]) < torch.finfo(torch.float32).eps


def domain_boundary(x):
    # unit square
    return torch.abs(x[0]) < torch.finfo(torch.float32).eps or torch.abs(x[1]) < torch.finfo(torch.float32).eps or\
           torch.abs(x[0]) > 1.0 - torch.finfo(torch.float32).eps or \
           torch.abs(x[1]) > 1.0 - torch.finfo(torch.float32).eps


def ess_boundary_fun(x):
    return 0.0


def flux(x):
    q = np.array([a[0] + a[2]*x[1], a[1] + a[2]*x[0]])
    return q


mesh.set_essential_boundary(origin, ess_boundary_fun)
mesh.set_natural_boundary(domain_boundary)

#Specify right hand side and stiffness matrix
rhs = PoissonFEM.RightHandSide(mesh)
rhs.set_natural_rhs(mesh, flux)
K = PoissonFEM.StiffnessMatrix(mesh)
rhs.set_rhs_stencil(mesh, K)

# define fom
fom = ROM.ROM(mesh, K, rhs, lin_dim_fom**2)
# Change for non unit square domains!!
xx, yy = torch.meshgrid((torch.linspace(0, 1, lin_dim_fom), torch.linspace(0, 1, lin_dim_fom)))
X = torch.cat((xx.flatten().unsqueeze(1), yy.flatten().unsqueeze(1)), 1)
fom.mesh.get_interpolation_matrix(X)
fom_autograd = fom.get_autograd_fun()


scale_tril = torch.cholesky(permeability_random_field.get_covariance_matrix())
mu_zero = torch.zeros(permeability_random_field.X.shape[0])


def joint_posterior():
    x = pyro.sample('x', dist.MultivariateNormal(mu_zero, scale_tril=scale_tril))
    lambdaf = torch.exp(x)
    uf = fom_autograd(lambdaf)
    uf_observed = pyro.sample('uf_observed', dist.Normal(uf, torch.ones_like(uf)/beta))
    return uf_observed


print('x == ', x := permeability_random_field.sample_log_permeability())
torch.save(x, 'x_true.pt')
print('uf_observed == ', uf_observed := fom_autograd(torch.exp(x)))
torch.save(uf_observed, 'uf_observed.pt')


def conditioned_posterior(uf_observed):
    return pyro.condition(joint_posterior, data={"uf_observed": uf_observed})


nuts_kernel = NUTS(conditioned_posterior(uf_observed))
mcmc = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=n_warmup, num_chains=n_chains)
mcmc.run()
mcmc.summary()
torch.save(mcmc.get_samples()['x'], 'x_samples.pt')


plt.plot(x[0])
plt.plot(torch.mean(mcmc.get_samples()['x'], 0))


fig = plt.figure(figsize=(15, 7))
ax = plt.subplot(1, 2, 1)
im0 = plt.imshow(x[0].view(lin_dim_fom, lin_dim_fom))
plt.colorbar(im0)
ax = plt.subplot(1, 2, 2)
im1 = plt.imshow(torch.mean(mcmc.get_samples()['x'], 0).view(lin_dim_fom, lin_dim_fom))
plt.colorbar(im1)
now = datetime.now()   # current date and time
plt.savefig(now.strftime("./%Y-%m-%d-%H-%M-%S") + f'inverse_problem_res={lin_dim_fom}')





