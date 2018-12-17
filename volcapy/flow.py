from volcapy import loading
import volcapy.matrix as mt
import volcapy.fast_covariance as ft

import numpy as np


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
n_cov_cols = coords.shape[0]

# COVARIANCE KERNEL PARAMS
sigma_2 = 200.0
lambda_2 = 200.0**2


# Prepare a function for returning partial rows of the covariance matrix.
def partial_covariance(row_begin, row_end):
    n_rows = row_end - row_begin + 1
    out = np.zeros((n_rows , n_cov_cols))
    return ft.build_cov(coords, out, row_begin, row_end)

r_begin = 0
r_end = 1000

# Perform the multiplication
t = mt.partial_mult(partial_covariance, F.T, r_begin, r_end)
