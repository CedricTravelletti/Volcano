""" This script aims at providing a fully gpytorch-based implementation of
training gaussian process models for inverse problems.

JUST FOR EASY USE WITH SLIME.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.covariance.matern32 as cl

import torch

import gpytorch
import gpytorch.lazy
from LBFGS import FullBatchLBFGS

import numpy as np
import os


checkpoint_size = 20000


# ----------------------------------------------------------------------------#
#      LOAD NIKLAS DATA
# ----------------------------------------------------------------------------#
# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
n_data = inverseProblem.n_data

# -- Delete Regularization Cells --
# Delete the cells.
# reg_cells_inds = get_regularization_cells_inds(inverseProblem)
# inverseProblem.forward[:, reg_cells_inds] = 0.0

F = torch.as_tensor(inverseProblem.forward).detach()
F = F.contiguous()

# Careful: we have to make a column vector here.
data_std = 0.1
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])

noise_var = torch.ones(d_obs.shape[0]) * data_std**2

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
del(inverseProblem)

train_x = cells_coords
train_y = d_obs

# ----------------------------------------------------------------------------
# Normalization and train/test split.
# ----------------------------------------------------------------------------
import numpy as np

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()

output_device = torch.device('cuda:0')

train_x, train_y = train_x.to(output_device), train_y.to(output_device)
F = F.to(output_device)

# ----------------------------------------------------------------------------
# Get GPUs.
# ----------------------------------------------------------------------------
n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

# ----------------------------------------------------------------------------
# Define model.
# ----------------------------------------------------------------------------
train_x.to(output_device)
mean_module = gpytorch.means.ConstantMean().to(output_device)
mean_module.initialize(constant=2000.0)

# base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
base_covar_module.initialize(outputscale=400.0**2)
base_covar_module.base_kernel.initialize(lengthscale=200.0)

covar_module = gpytorch.kernels.MultiDeviceKernel(
    base_covar_module, device_ids=range(n_devices),
    output_device=output_device
)
    
# Warning: we want to work with row vectors.
mean_x = torch.mm(F, mean_module(train_x)[:, None])[:, -1]
my_chunked_kernel = gpytorch.lazy.LazyEvaluatedKernelTensor(train_x,
    train_x, covar_module).to(output_device)

# TODO: Add caching.
with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
    cov_pushfwd = my_chunked_kernel._matmul(F.t())

covar_x = torch.mm(F, cov_pushfwd)

# Try computing (FC_mF^t+Delta_eps)^(-1) y via conjugate gradients.
from gpytorch.utils.linear_cg import linear_cg

size = 10000
matrix = K_d
K_tilde_d = matrix + torch.eye(matrix.size(-1),
        dtype=torch.float64).mul_(data_std**2).to(output_device)
solves = linear_cg(K_tilde_d.matmul, rhs=rhs.double(), max_iter=size)
