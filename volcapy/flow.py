from volcapy import loading
import volcapy.matrix as mt
import volcapy.inversion as inv
import volcapy.fast_covariance as ft

import numpy as np
from math import floor


# Globals
dx = 50
dy = 50
dz = 50
spacings = (dx, dy, dz)

# LOAD DATA
data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
data = loading.load_niklas(data_path)

coords = data['coords']
F = data['F']
d_obs = data['d']

# Dimensions
n_model = coords.shape[0]
n_data = F.shape[0]

# Build data covariance matrix.
sigma_d = 0.1
cov_d = np.diag([sigma_d] * n_data)

# COVARIANCE KERNEL PARAMS
sigma_2 = 200.0
lambda_2 = 200.0**2

# Prepare a function for returning partial rows of the covariance matrix.
def build_partial_covariance(row_begin, row_end):
    """

    Warning: should cast, since returns MemoryView.
    """
    n_rows = row_end - row_begin + 1
    out = np.zeros((n_rows , n_model))
    return ft.build_cov(coords, out, row_begin, row_end)

def compute_Cm_Gt(G):
    """ Compute the matrix product C_m * G^T.
    """
    n_data = G.shape[0]
    out = np.zeros((n_model, n_data))

    # Store the transpose once and for all.
    GT = G.T

    # Create the list of chunks.
    chunks = []
    for i in range(floor(n_model / 1000)):
        chunks.append((i * 1000, i * 1000 + 999))

    # Last chunk cannot be fully loop.
    chunks.append((floor(n_model / 1000.0)*1000, n_model - 1))

    # Loop in chunks of 1000.
    for row_begin, row_end in chunks:
        print(row_begin)
        # Get corresponding part of covariance matrix.
        partial_cov = build_partial_covariance(row_begin, row_end)

        # Append to result.
        out[row_begin:row_end + 1, :] = partial_cov @ GT

    return out


# Build prior mean.
# The order parameters are statically compiled in fast covariance.
m_prior = np.full(fl.n_model, 2300.0)

# Compute big matrix product and save.
out = fl.compute_Cm_Gt(fl.F)
np.save('Cm_Gt.npy', out)

# Use to perform inversion and save.
temp = fl.F @ out
inverse = np.linalg.inv(temp + fl.cov_d)

m_final = m_prior + out @ inverse @ (fl.d_obs - fl.F @ m_prior)
np.save('m_final.npy', m_final)
