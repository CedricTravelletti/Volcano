# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

LOAD THE COVARIANCE PUSHFORWARD, DO NOT COMPUTE IT.

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

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

# ----------------------------------------------------------------------------#
#      LOAD NIKLAS DATA
# ----------------------------------------------------------------------------#
# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
print(inverseProblem.data_values.shape)


# ----------------------------------------------------------------------------#
# SUBSETTING
# ----------------------------------------------------------------------------#
F_test, d_obs_test, inds = inverseProblem.subset_data(350)
F_test = torch.from_numpy(F_test)
d_obs_test = torch.from_numpy(d_obs_test)


n_data = inverseProblem.n_data
F = torch.as_tensor(inverseProblem.forward).detach()

# Careful: we have to make a column vector here.
data_std = 10.0
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
data_cov = torch.eye(n_data)
cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach()
del(inverseProblem)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
# Squared exp.
"""
sigma0_init = 200.0
m0 = 2200.0
lambda0 = 225.0

# Exp.
sigma0_init = 172.4
m0 = 2200.0
lambda0 = 102.0
# Matern 32
sigma0_init = 199.0
m0 = 2040.0
lambda0 = 475.0
"""

# Matern 52
sigma0_init = 199.0
m0 = 2068.0
lambda0 = 375.0

# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

###########
# IMPORTANT
###########
out_folder = "/idiap/temp/ctravelletti/out/profiles/"
"""
cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_225_sqexp.npy"
cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd"
cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_225_sqexp.npy"
cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_225_sqexp.npy"
"""
# cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_475_matern32.npy"
# cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_475_matern32.npy"
cov_pushfwd_path = "/home/cedric/PHD/run_results/forwards/cov_pushfwd_375_matern52.npy"

# Create the GP model.
myGP = GaussianProcess(F, d_obs, data_cov, sigma0_init)

sigma0s = np.arange(50, 500, 1)
n_sigma0s = len(sigma0s)

lls = np.zeros((n_sigma0s), dtype=np.float32)
train_rmses = np.zeros((n_sigma0s), dtype=np.float32)
loocv_rmses = np.zeros((n_sigma0s), dtype=np.float32)

def main(out_folder, lambda0, sigma0, cov_pushfwd_path):
    # Load the covariance pushforward.
    tmp = np.load(cov_pushfwd_path)
    print(tmp.shape)
    cov_pushfwd = torch.from_numpy(tmp[:, inds])

    # Run a forward pass.
    m_post_m, m_post_d = myGP.condition_model(
            cov_pushfwd, F, sigma0, concentrate=True)

    cond = myGP.condition_number(
            cov_pushfwd, F, sigma0)
    print("Condition number: {}".format(cond))

    ll = myGP.neg_log_likelihood().item()

    # Compute train_error
    train_rmse = myGP.train_RMSE().item()

    # Compute LOOCV RMSE.
    loocv_rmse = myGP.loo_error().item()

    # Compute test_error
    test_rmse = torch.sqrt(torch.mean(
        (d_obs_test - torch.mm(F_test, m_post_m))**2))
    
    test_rmse = test_rmse.item()

    print("Log-likelihood: {}".format(ll))
    print("Train RMSE: {}".format(train_rmse))
    print("Test RMSE: {}".format(test_rmse))
    print("Loocv RMSE: {}".format(loocv_rmse))

    # Save
    """
    np.save(os.path.join(out_folder, "sigma0s.npy"), sigma0s)
    np.save(os.path.join(out_folder, "lls.npy"), lls)
    np.save(os.path.join(out_folder, "train_rmses.npy"), train_rmses)
    np.save(os.path.join(out_folder, "loocv_rmses.npy"), loocv_rmses)
    """

if __name__ == "__main__":
    main(out_folder, lambda0, sigma0_init, cov_pushfwd_path)
