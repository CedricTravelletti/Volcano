# File: sparse_mesh.py, Author: Cedric Travelletti, Date: 11.03.2019.
""" Compactly supported Wendland Kernel.

"""
from volcapy.inverse.flow import InverseProblem
from volcapy.grid.sparsifier import Sparsifier
import numpy as np


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

inverseProblem.subset(50000)

# Create the sparsifier that will take care of ignoring cell couple that are
# too distant from each other.
sparsifier = Sparsifier(inverseProblem)
radius = 100.0

## ML part.
import tensorflow as tf

n_model = inverseProblem.n_model
F = tf.convert_to_tensor(inverseProblem.forward)

# Build the distance matrix as a sparse tensor.
dist_mesh_vals, dist_mesh_inds = sparsifier.compute_sparse_distance_mesh(radius)
dist_mesh_shape = np.array([n_model, n_model], dtype=np.int64)

# feed_dict = {dist_mesh_sparse: tf.SparseTensorValue(dist_mesh_inds, dist_mesh_vals, dist_mesh_shape)}

# Initialization variables.
m0 = 2200.0
data_std = 0.1
length_scale = 100.0
sigma = 20.0

# Prior parameters
m0 = tf.Variable(m0)
length_scale = tf.Variable(length_scale)
sigma = tf.Variable(sigma)

# Prior mean and data covariance.
m_prior = tf.scalar_mul(m0, tf.ones((inverseProblem.n_model, 1)))
data_cov = tf.scalar_mul(data_std**2, tf.eye(inverseProblem.n_data))

# Careful: we have to make a column vector here.
d_obs = tf.convert_to_tensor(inverseProblem.data_values[:, None])

# Compute the values of the non-zero elements in the covariance matrix.
# We operate on the list of non-zero values, not on the matrix.
# This is easier, since taking the exp of zero values would unzero them.
covariance_mat_vals = tf.scalar_mul(
    tf.pow(sigma, 2),
    tf.exp(dist_mesh_vals / tf.scalar_mul(2.0, tf.pow(length_scale, 2))))

# Now put this in a sparse matrix.
covariance_mat_sparse = tf.SparseTensor(
        dist_mesh_inds, covariance_mat_vals,
        dist_mesh_shape)

pushforward_cov = tf.sparse.matmul(
                covariance_mat_sparse,
                tf.transpose(F)
                )
inv_inversion_operator = tf.add(
        data_cov,
        tf.matmul(F, pushforward_cov))
inversion_operator = tf.linalg.inv(inv_inversion_operator)

# Need to do it this way, otherwise rounding errors kill everything.
log_det = tf.negative(
        tf.linalg.logdet(inversion_operator))

prior_misfit = tf.subtract(d_obs, tf.matmul(F, m_prior))

m_posterior = tf.add(
        m_prior,
        tf.matmul(
            tf.matmul(pushforward_cov, inversion_operator),
            prior_misfit))

# Maximum likelyhood estimator of posterior mean, given values
# of sigma and lambda. Obtained using the concentration formula.
log_likelyhood = tf.add(
      log_det,
      tf.matmul(
          tf.transpose(prior_misfit),
          tf.matmul(inversion_operator, prior_misfit)))

# Setup optimization
adam = tf.train.AdamOptimizer(learning_rate=5.0)
min_handle = adam.minimize(log_likelyhood, var_list=[m0, sigma])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    nr_train = 100
    for i in range(nr_train):
        tmp = sess.run(pushforward_cov)
        print(tmp)
        """
        sess.run(log_likelyhood)
        sess.run(min_handle)
        current_m0 = sess.run(m0)
        current_sigma = sess.run(sigma)
        print("m0 {}".format(current_m0))
        print("sigma {}".format(current_sigma))
        print("log likelyhood {}".format(sess.run(log_likelyhood)))
        """
