# File: first_draft.py, Author: Cedric Travelletti, Date: 14.02.2019.
""" First try at tensorflow implementation of Tarantola inversion.

Goal of this first try is to compute the product C_m GT.

THIS TIME WEW TRY TO DO IT CHUNKED.

"""
from volcapy.loading import load_niklas

import numpy as np
import tensorflow as tf


niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
GT = niklas_data['F']
GT = GT.T
GT = tf.convert_to_tensor(GT, np.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)

data_values = niklas_data['d']
data_coords = niklas_data['data_coords']

# TODO: This should be a placeholder, so that does not get directly loaded into
# graph everytime.
# xs = coords[:, 0][:, None]

chunk_size = 7500

# ---------------------------------------------------------------------
# ------ Build the cell distance differences for the covariance matrix.
# ---------------------------------------------------------------------
# Build the couples of all matchings of x with itself.
diff_xt, diff_x = np.meshgrid(coords[:chunk_size, 0], coords[:, 0], indexing='ij')

diff_yt, diff_y = np.meshgrid(coords[:chunk_size, 1], coords[:, 1], indexing='ij')
diff_zt, diff_z = np.meshgrid(coords[:chunk_size, 2], coords[:, 2], indexing='ij')

del(coords)

diffx = diff_x - diff_xt
diffy = diff_y - diff_yt
diffz = diff_z - diff_zt

del(diff_x)
del(diff_xt)
del(diff_y)
del(diff_yt)
del(diff_z)
del(diff_zt)

diff = np.stack(
        [diffx, diffy, diffz],
        axis=2)
# a = np.linalg.norm(diff, axis=2)
a = np.einsum('ijk,ijk->ij', diff, diff)
del(diff)
# ---------------------------------------------------------------------

# Parameters to optimize.
init_sigma_2 = 50.0**2
init_lambda_2 = 2 * 100.0**2

simga_2 = tf.Variable(init_sigma_2)
lambda_2 = tf.Variable(init_lambda_2)


place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))

b = tf.exp(-place)
out = tf.matmul(b, GT)

from timeit import default_timer as timer

with tf.Session() as sess:
    start = timer()
    print(sess.run(out, feed_dict={place: a}))
    end = timer()
    print(end - start)
