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


tf.logging.set_verbosity(tf.logging.DEBUG)

from volcapy.loading import load_niklas



# ------------------------------------------------------------------------
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
GT = niklas_data['F']
GT = GT.T

# SUBSTET
GT = GT[:50000, :]
GT = tf.convert_to_tensor(GT, np.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
# ------------------------------------------------------------------------

# SUBSET
coords = coords[:50000, :]

# Parameters to optimize.
init_sigma_2 = 50.0**2
init_lambda_2 = 2 * 100.0**2

simga_2 = tf.Variable(init_sigma_2)
lambda_2 = tf.Variable(init_lambda_2)


def per_line_operation(line):
    """ Operation to perform on each line.
    That is, given a coords line corresponding to one of the model cells,
    return the corresponding line of the matrix C_M GT.

    """
    diff_xt, diff_x = tf.meshgrid(line[0], coords[:, 0], indexing='ij')
    diff_yt, diff_y = tf.meshgrid(line[1], coords[:, 1], indexing='ij')
    diff_zt, diff_z = tf.meshgrid(line[2], coords[:, 2], indexing='ij')

    diffx = tf.math.squared_difference(diff_x, diff_xt)
    diffy = tf.math.squared_difference(diff_y, diff_yt)
    diffz = tf.math.squared_difference(diff_z, diff_zt)

    del(diff_x)
    del(diff_xt)
    del(diff_y)
    del(diff_yt)
    del(diff_z)
    del(diff_zt)

    diff = tf.add_n([diffx, diffy, diffz])
    b = tf.exp(
            tf.math.negative(tf.divide(diff, lambda_2)))
    out = tf.matmul(b, GT)
    return out

# ---------------------------------------------------------------------
final = tf.map_fn(per_line_operation, coords)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    a = sess.run(final)
    print(a)


place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))
