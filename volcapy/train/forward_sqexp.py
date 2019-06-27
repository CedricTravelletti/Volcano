# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.covariance_tools as cl

import numpy as np
import os

# Set up logging.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now torch in da place.
import torch
torch.set_num_threads(4)
# Choose between CPU and GPU.
device = torch.device('cuda:0')

# ----------------------------------------------------------------------------#
#      LOAD NIKLAS DATA
# ----------------------------------------------------------------------------#
# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
n_data = inverseProblem.n_data
F = torch.as_tensor(inverseProblem.forward).detach()

# Careful: we have to make a column vector here.
data_std = 0.1
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
data_cov = torch.mul(data_std**2, torch.eye(n_data))
cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
del(inverseProblem)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
sigma0_init = 100.0
m0 = 2000.0
lambda0 = 200.0
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

###########
# IMPORTANT
###########
out_folder = "/idiap/temp/ctravelletti/out/forwards/"

# Create the GP model.
myGP = GaussianProcess(F, d_obs, data_cov, sigma0_init)


def main(out_folder, lambda0, sigma0):
    # Create the covariance pushforward.
    cov_pushfwd = cl.compute_cov_pushforward(
            lambda0, F, cells_coords, gpu, n_chunks=200,
            n_flush=50):

    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.forward_model(
            cov_pushfwd, F, sigma0, concentrate=True)

    # Compute train_error
    train_error = model.train_error()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = model.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Save
    # filename = "m_post_" + str(lambda0) + "_matern.npy"
    # np.save(os.path.join(out_folder, filename), m_post_m)
