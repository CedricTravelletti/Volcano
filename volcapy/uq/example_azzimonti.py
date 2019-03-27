# File: azzimonti_run.py, Author: Cedric Travelletti, Date: 29.01.2019.
""" Run script for the Azzimonti techniques. It assumes posterior mean and
variance have already been computed and stored as static files.

"""
import numpy as np
from volcapy.loading import load_niklas
import volcapy.plotting.plot as plt

# Load the precomputed posterior mean and variance.
mean_path = "/home/cedric/PHD/Dev/Volcano/results/m_posterior.npy"
var_path = "/home/cedric/PHD/Dev/Volcano/results/posterior_cov_diag.npy"

mean = np.load(mean_path)
var = np.load(var_path)

# Load the dsm, so we can plot 3D results.
data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
data = load_niklas(data_path)
coords = data['coords']

# Select cells at 50m from bottom.
slice_inds = coords[:, 2] == 50.0

# Vorobev stuff.
import volcapy.uq.azzimonti as azz
mygp = azz.GaussianProcess(mean, var, covariance_func=None)

excu = mygp.compute_excursion_probs(threshold=2500.0)

# Plotting
import volcapy.plotting.plot as plt

