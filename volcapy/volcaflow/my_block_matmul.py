# File: first_draft.py, Author: Cedric Travelletti, Date: 14.02.2019.
""" First try at tensorflow implementation of Tarantola inversion.

Goal of this first try is to compute the product C_m GT.

THIS TIME WEW TRY TO DO IT CHUNKED.

"""
from volcapy.loading import load_niklas

import numpy as np
import tensorflow as tf


niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
F = niklas_data['F']
coords = niklas_data['coords']
data_values = niklas_data['d']
data_coords = niklas_data['data_coords']

# TODO: This should be a placeholder, so that does not get directly loaded into
# graph everytime.
xs = coords[:, 0][:, None]
F = tf.convert_to_tensor(F, np.float32)

chunk_size = 2000

# ---------------------------------------------------------------------
# ------ Build the cell distance differences for the covariance matrix.
# ---------------------------------------------------------------------
# Build the couples of all matchings of x with itself.
diff_xt, diff_x = tf.meshgrid(coords[:chunk_size, 0], coords[:, 0])
diff_yt, diff_y = tf.meshgrid(coords[:chunk_size, 1], coords[:, 1])
diff_zt, diff_z = tf.meshgrid(coords[:chunk_size, 2], coords[:, 2])

diff = tf.stack(
        [diff_x - diff_xt, diff_y - diff_yt, diff_z - diff_zt],
        axis=2)
a = tf.norm(diff, axis=2)
# ---------------------------------------------------------------------

# Parameters to optimize.
init_sigma_2 = 50.0**2
init_lambda_2 = 2 * 100.0**2

simga_2 = tf.Variable(init_sigma_2)
lambda_2 = tf.Variable(init_lambda_2)

"""
# Loop chunks.
for i in range(n_chunks):
    
    a = matrices[get_chunk_ind(i)]
    b = tf.exp(tf.pow(a, 2))
    
    # Partial multiplication.
    partial_mat_list[get_chunk_ind(i)] = tf.matmul(part, GT)
    
# ---------------------------------------------------------------------
# Time for the Reduce.
# Reducer is a stack operation along axis 0 for all the members of the partial
# matrix list.
# ---------------------------------------------------------------------
pushforward_cov = tf.stack(partial_mat_list, axis=0)
"""
GT = tf.transpose(F)

b = tf.exp(-tf.pow(a, 2))
out = tf.matmul(tf.transpose(b), GT)

from timeit import default_timer as timer

with tf.Session() as sess:
    start = timer()
    print(sess.run(out))
    end = timer()
    print(end - start)
