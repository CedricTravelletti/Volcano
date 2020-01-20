# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Try using the alternative conditioning formula from Tarantola.
It involves product with the inverse of the full covariance matrix, which is
now doable thanks to GPytorch.

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

from timeit import default_timer as timer

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

# Note we work with row vectors.
d_obs = torch.as_tensor(inverseProblem.data_values)

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
class ExactInverseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, F, likelihood, n_devices):
        super(ExactInverseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=2000.0)
        # base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        base_covar_module.initialize(outputscale=400.0**2)
        base_covar_module.base_kernel.initialize(lengthscale=200.0)
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
        self.F = F
    
    def forward(self, x):
        # Warning: we want to work with row vectors.
        mean_x = torch.mm(self.F, self.mean_module(x)[:, None])[:, -1]
        my_chunked_kernel = gpytorch.lazy.LazyEvaluatedKernelTensor(x,
            x, self.covar_module).to(output_device)

        # Compute covariance pushforward.
        # TODO: Add caching.
        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
            temp = my_chunked_kernel.inv_matmul(self.F.t())

            # Verify it gives the correct result.
            verif = my_chunked_kernel._matmul(temp)
            print("Verification.")
            print(verif - self.F.t())

        return temp


def train(train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter):

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_var, learn_additional_noise=False).to(output_device)
    model = ExactInverseGPModel(train_x, train_y, F, likelihood, n_devices).to(output_device)
    # model.train()


    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):
             return model(train_x)


start = timer()
res = train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=10000,
                          preconditioner_size=100,
                          n_training_iter=2000)

end = timer()
print(res)
print(end - start)

