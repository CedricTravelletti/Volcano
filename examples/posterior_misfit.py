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

d_obs = inverseProblem.data_values[:, None]
F = inverseProblem.forward

# Load posterior mean.
m_post_path = "/home/cedric/temp/m_posterior.npy"
m_posterior = np.load(m_post_path)[:, None]
