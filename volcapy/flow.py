from volcapy import loading
import volcapy.matrix_tools as mt
import volcapy.fast_covariance as ft

import numpy as np
from math import floor


# Globals
sigma_2 = 50.0**2
lambda_2 = 130**2

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


# Prepare a function for returning partial rows of the covariance matrix.
def build_partial_covariance(row_begin, row_end):
    """
    Warning: should cast, since returns MemoryView.
    """
    n_rows = row_end - row_begin + 1
    out = np.zeros((n_rows , n_model))
    return ft.build_cov(coords, out, row_begin, row_end,
            sigma_2, lambda_2)

# TODO: Refactor. Effectively, this is chunked multiplication of a matrix with
# an implicitly defined one.
def compute_Cm_Gt(G):
    """ Compute the matrix product C_m * G^T.
    """
    n_data = G.shape[0]
    out = np.zeros((n_model, n_data))

    # Store the transpose once and for all.
    GT = G.T

    # Create the list of chunks.
    chunk_size = 1000
    chunks = []
    for i in range(floor(n_model / chunk_size)):
        chunks.append((i * chunk_size, i * chunk_size + chunk_size - 1))

    # Last chunk cannot be fully loop.
    chunks.append((floor(n_model / float(chunk_size))*chunk_size, n_model - 1))

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
m_prior = np.full(n_model, 2350.0)

# Compute big matrix product and save.
print("Computing big matrix product.")
out = compute_Cm_Gt(F)
np.save('Cm_Gt.npy', out)

# Use to perform inversion and save.
print("Inverting matrix.")
temp = F @ out
inverse = np.linalg.inv(temp + cov_d)

# TODO: Warning, compensating for Bouguer anomaly.
print("Computing posterior mean.")
m_posterior = m_prior + out @ inverse @ (d_obs - F @ m_prior)
np.save('m_posterior.npy', m_posterior)

# ----------------------------------------------
# Build and save the diagonal of the posterior covariance matrix.
# ----------------------------------------------
print("Computing posterior variance.")
Cm_post = np.empty([n_model], dtype=out.dtype)
A = out @ inverse
B = out.T

# Diagonal is just given by the scalar product of the two factors.
for i in range(n_model):
    Cm_post[i] = np.dot(A[i, :], B[:, i])

# Save the square root standard deviation).
np.save("posterior_cov_diag.npy", np.sqrt(np.array([sigma_2]*n_model) - Cm_post))

# AMBITIOUS: Compute the whole (38GB) posterior covariance matrix.
print("Computing posterior covariance.")
post_cov = np.memmap('post_cov.npy', dtype='float32', mode='w+',
        shape=(n_model, n_model))

# Compute the matrix product line by line.
for i in range(n_model):
    print(i)
    prior_cov = build_partial_covariance(i, i)
    post_cov[i, :] = prior_cov - A[i, :] @ B

# Flush to disk.
del post_cov
