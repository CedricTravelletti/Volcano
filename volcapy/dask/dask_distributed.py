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
import numpy as np
import dask.array as da
import os
from timeit import default_timer as timer

from volcapy.loading import load_niklas
# ------------------------------------------------------------------------

# Connect to scheduler.
from distributed import Client
client = Client("172.19.103.8:8786")


niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"

# Load data from Niklas (forward and measurements).
niklas_data = load_niklas(niklas_data_path)


F = niklas_data['F']
n_data = F.shape[0]
n_model = F.shape[1]
model_chunk_size = 1000
F = da.from_array(F, chunks=(n_data, model_chunk_size))

import psutil
process = psutil.Process(os.getpid())


d_obs = niklas_data['d']
d_obs = da.from_array(d_obs, chunks=1000)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
coords_x = np.ascontiguousarray(coords[:,0])
coords_y = np.ascontiguousarray(coords[:,1])
coords_z = np.ascontiguousarray(coords[:,2])


coords_x = da.from_array(coords_x, chunks=1000)
coords_y = da.from_array(coords_y, chunks=1000)
coords_z = da.from_array(coords_z, chunks=1000)

print(str(process.memory_info().rss / 1e9))

x1, x2 = da.meshgrid(coords_x, coords_x, indexing='ij')
y1, y2 = da.meshgrid(coords_y, coords_y, indexing='ij')
z1, z2 = da.meshgrid(coords_z, coords_z, indexing='ij')

print(str(process.memory_info().rss / 1e9))

diff_x = da.subtract(x1, x2)
diff_y = da.subtract(y1, y2)
diff_z = da.subtract(z1, z2)

squared_x = da.square(diff_x)
squared_y = da.square(diff_y)
squared_z = da.square(diff_z)

inv_lambda_2 = - 1/(130.0**2)


exp_x = da.exp(inv_lambda_2, squared_x)
exp_y = da.exp(inv_lambda_2, squared_y)
exp_z = da.exp(inv_lambda_2, squared_z)

print(str(process.memory_info().rss / 1e9))


sigma_2 = 50.0**2
exp_tot = da.multiply(sigma_2,
        da.multiply(exp_x,
        da.multiply(exp_y, exp_z)))

pushforward_cov = exp_tot.dot(F.transpose())

start = timer()

final = pushforward_cov[:100000, :1000].compute()

end = timer()
print(final)
print(str((end - start)/60.0))
print(pushforward_cov.shape)

"""
# Data covariance matrix.
sigma_d = 0.1
cov_d = sigma_d**2 * tf.eye(nr_data, dtype=tf.float32)

coords = niklas_data['coords']
coords = coords.astype(np.float32, copy=False)
# coords = tf.convert_to_tensor(coords)

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


inversion_operator = tf.matmul(
        pushforward_cov,
        tf.linalg.inv(tf.add(tf.matmul(F, pushforward_cov), cov_d))
        )

prior_misfit = tf.subtract(d_obs, tf.matmul(F, m_prior))
m_posterior = tf.add(m_prior, tf.matmul(inversion_operator, prior_misfit))
d_posterior = tf.matmul(F, m_posterior)
        

mse = tf.losses.mean_squared_error(d_posterior, d_obs)
adam = tf.train.AdamOptimizer(learning_rate=5.0)
min_handle = adam.minimize(mse, var_list=mu_0)
# place = tf.placeholder(tf.float32, shape=(chunk_size, GT.shape[0]))

my_mu0 = []
my_lambda = []
nr_train = 3

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(nr_train):
        sess.run(min_handle)
        my_mu0.append(sess.run(mu_0))
        my_lambda.append(sess.run(lambda_2))
    sess.run(pushforward_cov)
    end = timer()
    # a = sess.run(d_posterior)
print(str((end - start)/60.0))
"""
