""" Presentation for meeting with Niklas 13 February.

"""
from volcapy.loading import load_niklas
import volcapy.plotting.plot as plt
import volcapy.uq.azzimonti as azz

import numpy as np


# posterior_mean_path = "/home/cedric/PHD/Dev/Volcano/results/m_posterior.npy"
posterior_mean_path = "/home/cedric/PHD/run_results/forwards/m_post_902_exponential.npy"
posterior_mean_path2 = "/home/cedric/PHD/run_results/forwards/m_post_342_squared_exponential.npy"

"""
posterior_mean_path = "/home/cedric/PHD/Dev/Volcano/results/m_posterior.npy"
posterior_variance_path = "/home/cedric/PHD/Dev/Volcano/results/posterior_cov_diag.npy"
"""
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

# Load inversion results.
post_mean = np.load(posterior_mean_path)
post_mean2 = np.load(posterior_mean_path2)
# post_variance = np.load(posterior_variance_path)

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
F = niklas_data['F']
coords = niklas_data['coords']
data_values = niklas_data['d']
data_coords = niklas_data['data_coords']

# ---------------------
# Inversion parameters: mean = 2350, lambda = sigma_2 = 50.0**2, lambda_2 =
# 130**2. Data noise = 0.1**2.
# ---------------------

# Island boundaries.
island_minx = 516000.0
island_maxx = 522000.0
island_miny = 4291000.0
island_maxy = 4296000.0

# Find the indices of the cells within the island.
island_inds = np.where(
        (coords[:, 0] > island_minx)
        & (coords[:, 0] < island_maxx)
        & (coords[:, 1] > island_miny)
        & (coords[:, 1] < island_maxy))[0]

# Out of these, find the surface cells (roughly, those are the first 32000).
surface_inds = island_inds[island_inds < 32000]

# Plot surface.
plt.plot_region(surface_inds, post_mean.reshape(-1), coords, cmin=2250, cmax=2550.0)
plt.plot_region(surface_inds, post_mean2.reshape(-1), coords, cmin=2100, cmax=2550.0)

plt.plot_region(island_inds, post_mean2.reshape(-1), coords, cmin=2200,
        cmax=2500.0, n_sample=10000)

plt.plot(post_mean2.reshape(-1), coords, cmin=2100, cmax=2550.0, n_sample=50000)

plt.plot_z_slice([-500, 0, 500], post_mean.reshape(-1), coords, cmin=1800, cmax=2550.0)
plt.plot_z_slice([-500, 0, 500], post_mean2.reshape(-1), coords)

# Plot slices as in Niklas.
plt.plot_z_slice(0.0, post_mean,
        coords[:, 0],coords[:, 1],coords[:, 2])

plt.plot_z_slice(500.0, post_mean,
        coords[:, 0],coords[:, 1],coords[:, 2])

plt.plot_z_slice(800.0, post_mean,
        coords[:, 0],coords[:, 1],coords[:, 2])

plt.plot_z_slice(-100.0, post_mean,
        coords[:, 0],coords[:, 1],coords[:, 2])

# Plot excursion set plug-in estimate.
excursion_inds = np.where(post_mean > 2500.0)
plt.plot_region(excursion_inds, post_mean,
        coords[:, 0],coords[:, 1],coords[:, 2])

# Plot residual standard deviation.
post_std_dev = np.sqrt(post_variance)
plt.plot_region(island_inds, post_std_dev,
        coords[:, 0],coords[:, 1],coords[:, 2])

# Plot a slice of the residual std at sea level.
plt.plot_z_slice(0.0, post_std_dev,
        coords[:, 0],coords[:, 1],coords[:, 2])

# Plot individual measurement.
# Index of the measurement we want to consider.
data_index = 200

my_coords = np.vstack((coords, data_coords[data_index]))

my_mean = np.append(post_mean, 100000)

plt.plot_region(my_surface_inds, my_mean, my_coords[:,0], my_coords[:,1],
        my_coords[:,2])


# Azzimonti Stuff
mygp = azz.GaussianProcess(post_mean, post_variance, covariance_func=None)
excu = mygp.compute_excursion_probs(threshold=2500.0)

# Vorobev expectation.
vorb_inds = mygp.vorobev_expectation_inds(threshold=2500.0)
plt.plot_region(vorb_inds, excu, coords[:,0], coords[:,1], coords[:,2])
