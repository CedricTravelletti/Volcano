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

import tempfile
from tensorflow.python.client import timeline


tf.logging.set_verbosity(tf.logging.DEBUG)

from volcapy.loading import load_niklas



# ------------------------------------------------------------------------
niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)
GT = niklas_data['F']
GT = GT.T

# SUBSTET
GT = GT[:120000, :]
GT = tf.convert_to_tensor(GT, np.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
# ------------------------------------------------------------------------

# SUBSET
coords = coords[:120000, :]

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

# ---------------------------------------------------------------------
final = tf.map_fn(per_line_operation, coords)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    a = sess.run(final, options=options, run_metadata=run_metadata)

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()

    with open('timeline_02_step.json', 'w') as f:
        f.write(chrome_trace)


# place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))
