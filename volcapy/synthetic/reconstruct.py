# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

"""
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.covariance_tools as cl

import numpy as np
import os


nx = 50
ny = 50
nz = 50
res_x = 1
res_y = 1
res_z = 1


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
#      LOAD DATA
# ----------------------------------------------------------------------------#
data_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/out/"
cells_coords = np.load(os.path.join(data_folder, "coords_synth.npy"))
data_values = np.load(os.path.join(data_folder, "data_values_synth.npy"))
F = np.load(os.path.join(data_folder, "F_synth.npy"))

n_data = data_values.shape[0]

# Careful: we have to make a column vector here.
data_std = 0.1

d_obs = data_values.astype(np.float32)
cells_coords = cells_coords.astype(np.float32)
F = F.astype(np.float32)

d_obs = torch.as_tensor(data_values[:, None]).float()
cells_coords = torch.as_tensor(cells_coords).detach().float()
F = torch.as_tensor(F).float()

data_cov = torch.eye(n_data, dtype=torch.float32)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
sigma0_init = 200.0
m0 = 2200.0
lambda0 = 100.0
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

###########
# IMPORTANT
###########
out_folder = "/home/cedric/PHD/Dev/Volcano/volcapy/synthetic/forwards"

# Create the GP model.
data_std = 1000.0
myGP = GaussianProcess(F, d_obs, data_cov, sigma0_init,
        data_std)


def main(out_folder, lambda0, sigma0):
    # Create the covariance pushforward.
    cov_pushfwd = cl.compute_cov_pushforward(
            lambda0, F, cells_coords, cpu, n_chunks=200,
            n_flush=50)
    K_d = torch.mm(F, cov_pushfwd)

    """
    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)
    """
    m_post_d = myGP.condition_data(
            K_d, sigma0, concentrate=True)

    # Compute diagonal of posterior covariance.
    post_cov_diag = myGP.compute_post_cov_diag(
            cov_pushfwd, cells_coords, lambda0, sigma0, cl)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Once finished, run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)

    # Compute train_error
    train_error = myGP.train_RMSE()

    logger.info("Train error: {}".format(train_error.item()))

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error()
    logger.info("LOOCV error: {}".format(loocv_rmse.item()))

    # Save
    filename = "m_post_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), m_post_m)

    filename = "post_cov_diag_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), post_cov_diag)

    filename = "cov_pushfwd_" + str(int(lambda0)) + "_sqexp.npy"
    np.save(os.path.join(out_folder, filename), cov_pushfwd)

    # ---------------------------------------------
    # A AMELIORER
    # ---------------------------------------------
    # Save to VTK format..
    from volcapy.synthetic.vtkutils import save_vtk

    save_vtk(m_post_m.numpy(), (nx, ny, nz), res_x, res_y, res_z, "test.mhd")

if __name__ == "__main__":
    main(out_folder, lambda0, sigma0_init)
