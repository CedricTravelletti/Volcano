# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.matern32 as cl

import numpy as np
import os

# Set up logging.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

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
data_cov = torch.eye(n_data)
cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
del(inverseProblem)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

NtV_crit = (0.1/200.0)**2




# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
sigma0_init = 199.0
m0 = 2200.0
lambda0 = 475.0
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
            n_flush=50)
    K_d = torch.mm(F, cov_pushfwd)

    """
    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)
    """
    m_post_d = myGP.condition_data(
            K_d, sigma0, concentrate=True,
            NtV_crit=NtV_crit)

    # Compute diagonal of posterior covariance.
    post_cov_diag = myGP.compute_post_cov_diag(
            cov_pushfwd, cells_coords, lambda0, myGP.sigma0, cl)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True,
            NtV_crit=NtV_crit)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Save
    filename = "m_post_" + str(int(lambda0)) + "_matern32.npy"
    np.save(os.path.join(out_folder, filename), m_post_m)

    filename = "post_cov_diag_" + str(int(lambda0)) + "_matern32.npy"
    np.save(os.path.join(out_folder, filename), post_cov_diag)

    filename = "cov_pushfwd_" + str(int(lambda0)) + "_matern32.npy"
    np.save(os.path.join(out_folder, filename), cov_pushfwd)

if __name__ == "__main__":
    main(out_folder, lambda0, sigma0_init)
