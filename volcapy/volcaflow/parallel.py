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
niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
GT = niklas_data['F']
GT = GT.T

# SUBSTET
divide = 50000
# GT1 = GT[divide:, :]
GT1 = GT
GT1 = tf.convert_to_tensor(GT1, np.float32)

# GT2 = GT[divide:, :]
GT2 = GT
GT2 = tf.convert_to_tensor(GT2, np.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
# ------------------------------------------------------------------------

# SUBSET
coords1 = coords[:divide, :]
coords2 = coords[divide:, :]

# Parameters to optimize.
init_sigma_2 = 50.0**2
init_lambda_2 = 130.0**2

sigma_2 = tf.Variable(init_sigma_2)
lambda_2 = tf.Variable(init_lambda_2)


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
    out = tf.matmul(b, GT1)
    return out


def per_line_operation2(line):
    """ Operation to perform on each line.
    That is, given a coords line corresponding to one of the model cells,
    return the corresponding line of the matrix C_M GT.

    """
    with tf.device('/cpu:0'):
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
        out = tf.matmul(b, GT2)
        return out

# ---------------------------------------------------------------------
final1 = tf.map_fn(per_line_operation, coords1)
with tf.device('/cpu:0'):
    final2 = tf.map_fn(per_line_operation2, coords2)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

start = timer()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(tf.concat([final1, final2], axis=0))

end = timer()
print(str((end - start)/60.0))


# place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))
