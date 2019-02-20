""" Block matrix operations in TensorFlow.
Cpoied from

http://pages.cs.wisc.edu/~akella/CS838/F16/assignment3/uploaded-scripts/exampleMatmulSingle.py

"""
"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""
import tensorflow as tf
import numpy as np
import os

from timeit import default_timer as timer

tf.logging.set_verbosity(tf.logging.DEBUG)


from volcapy.loading import load_niklas
# ------------------------------------------------------------------------
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)

# Data subset.
nr_data = 8

F = niklas_data['F']
F = F[:nr_data, :]
n_model = F.shape[1]
F = tf.convert_to_tensor(F, tf.float32)

GT = niklas_data['F']
GT = GT.T
GT = GT[:, :nr_data]
GT = tf.convert_to_tensor(GT, np.float32)

d_obs = niklas_data['d']
d_obs = d_obs[:nr_data][:, None]
d_obs = tf.convert_to_tensor(d_obs)

# Data covariance matrix.
sigma_d = 0.1
cov_d = sigma_d**2 * tf.eye(nr_data, dtype=tf.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
coords = tf.convert_to_tensor(coords)

# SUBSET THE MODEL.
coords_subset = tf.convert_to_tensor(coords[:, :])

# Parameters to optimize.
init_mu_0 = 2300.0
init_sigma_2 = 50.0**2
init_lambda_2 = 130.0**2

sigma_2 = tf.Variable(init_sigma_2)
lambda_2 = tf.Variable(init_lambda_2, trainable=True)
mu_0 = tf.Variable(init_mu_0, trainable=True)

# Build prior mean.
m_prior = tf.scalar_mul(mu_0, tf.ones([n_model, 1], dtype=tf.float32))


def per_line_operation(line):
    """ Operation to perform on each line.
    That is, given a coords line corresponding to one of the model cells,
    return the corresponding line of the matrix C_M GT.

    """
    diff_xt, diff_x = tf.meshgrid(line[0], coords[:, 0], indexing='ij')
    diff_yt, diff_y = tf.meshgrid(line[1], coords[:, 1], indexing='ij')
    diff_zt, diff_z = tf.meshgrid(line[2], coords[:, 2], indexing='ij')

    diffx = tf.squared_difference(diff_x, diff_xt)
    diffy = tf.squared_difference(diff_y, diff_yt)
    diffz = tf.squared_difference(diff_z, diff_zt)

    del(diff_x)
    del(diff_xt)
    del(diff_y)
    del(diff_yt)
    del(diff_z)
    del(diff_zt)

    diff = tf.add_n([diffx, diffy, diffz])
    b = tf.scalar_mul(sigma_2,
            tf.exp(tf.negative(tf.divide(diff, lambda_2))))
    out = tf.matmul(b, GT)
    return out


def per_chunk_operation(chunk):
    """ Operation to perform on each line.
    That is, given a coords line corresponding to one of the model cells,
    return the corresponding line of the matrix C_M GT.

    """
    diff_xt, diff_x = tf.meshgrid(chunk[:, 0], coords[:, 0], indexing='ij')
    diff_yt, diff_y = tf.meshgrid(chunk[:, 1], coords[:, 1], indexing='ij')
    diff_zt, diff_z = tf.meshgrid(chunk[:, 2], coords[:, 2], indexing='ij')

    diffx = tf.squared_difference(diff_x, diff_xt)
    diffy = tf.squared_difference(diff_y, diff_yt)
    diffz = tf.squared_difference(diff_z, diff_zt)

    del(diff_x)
    del(diff_xt)
    del(diff_y)
    del(diff_yt)
    del(diff_z)
    del(diff_zt)

    diff = tf.add_n([diffx, diffy, diffz])
    b = tf.scalar_mul(sigma_2,
            tf.exp(tf.negative(tf.divide(diff, lambda_2))))
    out = tf.matmul(b, GT)
    return out


# Compute C_M GT.
pushforward_cov = tf.squeeze(
        tf.map_fn(per_line_operation, coords_subset))
inversion_operator = tf.matmul(
        pushforward_cov,
        tf.linalg.inv(tf.add(tf.matmul(F, pushforward_cov), cov_d))
        )

prior_misfit = tf.subtract(d_obs, tf.matmul(F, m_prior))

d_posterior = tf.matmul(F,
        tf.add(m_prior, tf.matmul(inversion_operator, prior_misfit)))

mse = tf.losses.mean_squared_error(d_posterior, d_obs)
adam = tf.train.AdamOptimizer(learning_rate=5.0)
min_handle = adam.minimize(mse, var_list=mu_0)
# place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))

my_mu0 = []
my_lambda = []
nr_train = 3
with tf.Session() as sess:
    start = timer()
    sess.run(tf.global_variables_initializer())
    for i in range(nr_train):
        sess.run(min_handle)
        my_mu0.append(sess.run(mu_0))
        my_lambda.append(sess.run(lambda_2))
    end = timer()
    # a = sess.run(d_posterior)
