""" This script aims at providing a fully gpytorch-based implementation of
training gaussian process models for inverse problems.

01.04.2020: Copied from train_rmse, but uses mll insteads
This is our new start.
I think that the problems we were encountering before pausing development (i.e.
optimizer remaining stuck) are due to the default starting point of the
optimizer being in a flat zone.

Now that we have discovered how to setup starting values (thanks to tutoring),
we can try again.

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

        # The below shows how to have several different lengthscales.
        # self.base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

        # Don't know if that really works.
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

        # TODO: Add caching.
        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
            cov_pushfwd = my_chunked_kernel._matmul(self.F.t())

        covar_x = torch.mm(self.F, cov_pushfwd)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter):
    # Likelihood defines mapping of process values to observed data,
    # i.e. y = f(x) + eps. Here we know the noise so don't learn it.
    # Note that noise variance (not std) should be given.
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_var, learn_additional_noise=False).to(output_device)

    model = ExactInverseGPModel(train_x, train_y, F, likelihood, n_devices).to(output_device)
    model.train()

    # optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Starting values for optimization.
    model.mean_module.constant.data = torch.FloatTensor([3000.0])
    model.covar_module.base_kernel._set_outputscale(300**2)
    model.covar_module.base_kernel.base_kernel._set_lengthscale([200.0])

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):

        """
        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()
        """

        for i in range(n_training_iter):
            # options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            # loss, _, _, _, _, _, _, fail = optimizer.step(options)
            optimizer.zero_grad()
            output = model(train_x)

            # Compute train RMSE
            train_RMSE = torch.sqrt(torch.mean(torch.pow(output.loc - train_y, 2)))

            # loss = train_RMSE
            loss = -mll(output, train_y)
            # marg_log_lkl = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            print('Train RMSE: %.6f' % (train_RMSE.item()))
            # print('Marginal log-likelihood: %.6f' % (marg_log_lkl.item()))

            print('Iter %d/%d - Loss: %.6f   lengthscale: %.6f   variance: %.6f   mean: %.6f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                torch.sqrt(model.covar_module.module.outputscale.item()),
                model.mean_module.constant.item()
            ))

            """
            if fail:
                print('Convergence reached!')
                break
            """

    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


# ----------------------------------------------------------------------------
# Determine GPU settings.
# ----------------------------------------------------------------------------
import gc

def find_best_gpu_setting(train_x,
                          train_y,
                          n_devices,
                          output_device,
                          preconditioner_size
):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(train_x, train_y,
                         n_devices=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size

# Set a large enough preconditioner size to reduce the number of CG iterations run
preconditioner_size = 100
"""
checkpoint_size = find_best_gpu_setting(train_x, train_y,
                                        n_devices=n_devices,
                                        output_device=output_device,
                                        preconditioner_size=preconditioner_size)

"""
# ----------------------------------------------------------------------------
# Train.
# ----------------------------------------------------------------------------
model, likelihood = train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=10000,
                          preconditioner_size=100,
                          n_training_iter=2000)
