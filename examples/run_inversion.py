from volcapy.inverse.flow import InverseProblem
import numpy as np

# GLOBALS
data_std = 0.1
sigma_d = data_std
sigma0 = 88.95
lambda0 = 200.0
prior_mean = 2200.0

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

out_folder = "/home/cedric/temp/"
inverseProblem.inverse(out_folder, prior_mean, sigma_d,
        sigma0, lambda0,
        preload_covariance_pushforward=False, cov_pushforward=None,
        compute_post_covariance=False)
