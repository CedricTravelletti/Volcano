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
GT = niklas_data['F']
GT = GT.T

# SUBSET
GT = tf.convert_to_tensor(GT, np.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
coords = tf.convert_to_tensor(coords)
# ------------------------------------------------------------------------

# SUBSET
coords_subset = tf.convert_to_tensor(coords[:10, :])

# SPLITS
# splits = 17917*[10] + [1]
# coords_split = tf.split(coords, splits)
splits = 2*[5]
coords_split = tf.split(coords_subset, splits)

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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


final = tf.map_fn(per_line_operation, coords_subset)
start = timer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(final)

end = timer()
time_line = end - start
print("Per line.")
print(str((end - start)/60.0))


final2 = per_chunk_operation(coords_subset)
start = timer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(final2)

end = timer()
print("Per chunk.")
print(str((end - start)/60.0))
print("Line over chunk: " + str(time_line/(end - start)))


# SPLITS
# DOESNT WORK BECAUSE OF LIST UNPACKING.
# NOT WORTH PURSUING FOR THE MOMENT.
"""
final_splits = tf.map_fn(per_chunk_operation, coords_split)
start = timer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(final_splits)

end = timer()
print("Per splits.")
print(str((end - start)/60.0))
"""
