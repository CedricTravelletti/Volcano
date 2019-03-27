# File: sparse_mesh.py, Author: Cedric Travelletti, Date: 11.03.2019.
""" Sparse TensorFlow version of volcano code.
Goal is to sparsify the covariance matrix by ignoring cell couples where the
distance between the two cells is bigger than some radius. In that case, put
the covariance to 0.

Nieghbors-with-radius computations are done with a BallTree, which provides
really fast queries.

Run maximum likelyhood parameter estimation. Use concentrated version to avoid
optimizing the prior mean.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

from sklearn.neighbors import BallTree


# GLOBALS
m0 = 2071.0
data_std = 0.1

length_scale = 100.0
sigma = 20.0


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

inverseProblem.subset(50000)

tree = BallTree(inverseProblem.cells_coords)

def get_cells_ind_with_radius(cell, radius):
    """ Given a BallTree (based on an underlying list of cells), get indices of
    all the cells that are with radius of the given cell (actually, can be any
    point, not necessarily a cell of the BallTree).

    Parameters
    ----------
    cell: List[float, float, float]
        Coordinate of the point to consider.
    radius: float

    Returns
    -------
    List[int]
        Indices of the cells that are within the given radius.

    """
    return tree.query_radius(cell.reshape(1, -1), radius)[0]

def compute_sparse_distance_mesh(radius):
    # Lists to store the computed covariances and indices in the big sparse
    # matrix.
    covariances = []
    inds = []

    # Loop over all cells.
    for cell_ind, cell in enumerate(inverseProblem.cells_coords):
        print(cell_ind)
        # Get neighbors.
        neighbor_inds = get_cells_ind_with_radius(cell, radius)

        # Loop over all neighboring cells.
        for neighbor_ind in neighbor_inds:
            covariances.append(
                np.linalg.norm(
                    cell - inverseProblem.cells_coords[neighbor_ind, :]))
            inds.append((cell_ind, neighbor_ind))
    return (covariances, inds)

radius = 250.0
distance_mesh_vals, distance_mesh_inds = compute_sparse_distance_mesh(radius)

## ML part.
import tensorflow as tf


n_model = inverseProblem.n_model

# Convert inputs to tensors.
distance_mesh_vals = tf.convert_to_tensor(distance_mesh_vals)
distance_mesh_inds = tf.convert_to_tensor(distance_mesh_inds, dtype=np.int64)
F = tf.convert_to_tensor(inverseProblem.forward)

m0 = 2200.0
data_std = 0.1

length_scale = 100.0
sigma = 20.0

# Prior parameters
m0 = tf.Variable(m0)
length_scale = tf.Variable(length_scale)
sigma = tf.Variable(sigma)

# Prior mean
m_prior = tf.scalar_mul(m0, tf.ones((inverseProblem.n_model, 1)))

data_cov = tf.scalar_mul(data_std**2, tf.eye(inverseProblem.n_data))

# Careful: we have to make a column vector here.
d_obs = tf.convert_to_tensor(inverseProblem.data_values[:, None])

# Compute the values of the non-zero elements in the covariance matrix.
covariance_vals = tf.scalar_mul(
    tf.pow(sigma, 2),
    tf.exp(
        tf.div(
            distance_mesh_vals,
            tf.scalar_mul(2.0, tf.pow(length_scale, 2))
            )
        )
    )
# Build the covariance matrix as a sparse tensor.
covariance_mat = tf.SparseTensor(
        distance_mesh_inds, covariance_vals,
        dense_shape=[n_model, n_model])

pushforward_cov = tf.sparse.matmul(
                covariance_mat,
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
            prior_misfit)
        )

# Maximum likelyhood estimator of posterior mean, given values
# of sigma and lambda. Obtained using the concentration formula.
log_likelyhood = tf.add(
      log_det,
      tf.matmul(
          tf.transpose(prior_misfit),
          tf.matmul(inversion_operator, prior_misfit)
          )
      )


# Setup optimization
adam = tf.train.AdamOptimizer(learning_rate=5.0)
min_handle = adam.minimize(log_likelyhood, var_list=[m0, sigma])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    nr_train = 100
    for i in range(nr_train):
        sess.run(min_handle)
        current_m0 = sess.run(m0)
        current_sigma = sess.run(sigma)
        print(current_m0)
        print(current_sigma)
