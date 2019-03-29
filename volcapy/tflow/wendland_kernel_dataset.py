# File: sparse_mesh.py, Author: Cedric Travelletti, Date: 11.03.2019.
""" Compactly supported Wendland Kernel.

"""
from volcapy.inverse.flow import InverseProblem
from volcapy.grid.sparsifier import Sparsifier
import numpy as np
import tensorflow as tf


# BATCHING.
data_batch_size = 20


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# inverseProblem.subset(5000)

# Train-validation split.
nr_train = 450
F_valid_raw, d_obs_valid_raw = inverseProblem.subset_data(nr_train)

n_model = inverseProblem.n_model
F_train_raw = inverseProblem.forward
d_obs_train_raw = inverseProblem.data_values[:, None]

# Create the sparsifier that will take care of ignoring cell couple that are
# too distant from each other.
sparsifier = Sparsifier(inverseProblem)
radius = 200.0

# Build the distance matrix as a sparse tensor.
dist_mesh_vals, dist_mesh_inds = sparsifier.compute_sparse_distance_mesh(radius)
dist_mesh_vals = tf.convert_to_tensor(dist_mesh_vals, dtype=tf.float32)
dist_mesh_shape = np.array([n_model, n_model], dtype=np.int64)


# Initialization variables.
m0 = 2184.0
data_std = 0.1
length_scale = 50.0
sigma = 140.0

min_len = 0.0
max_len = 100.0

# Prior parameters
m0 = tf.Variable(m0)
length_scale = tf.Variable(length_scale)
clipped_length_scale = tf.clip_by_value(length_scale, min_len, max_len)
sigma = tf.Variable(sigma)

# Prior mean and data covariance.
m_prior = tf.scalar_mul(m0, tf.ones((inverseProblem.n_model, 1)))
data_cov = tf.scalar_mul(data_std**2, tf.eye(data_batch_size))


# Compute the values of the non-zero elements in the covariance matrix.
# We operate on the list of non-zero values, not on the matrix.
# This is easier, since taking the exp of zero values would unzero them.
covariance_mat_vals = tf.scalar_mul(
    tf.pow(sigma, 2),
    tf.multiply(
        tf.pow(tf.nn.relu((1 - tf.div(dist_mesh_vals, 2 *
                clipped_length_scale))),4),
        (1 + tf.div(2 * dist_mesh_vals, clipped_length_scale))))

# Now put this in a sparse matrix.
covariance_mat_sparse = tf.SparseTensor(
        dist_mesh_inds, covariance_mat_vals,
        dist_mesh_shape)

# Setup datasets.
F_train = tf.data.Dataset.from_tensor_slices(F_train_raw)
d_obs_train = tf.data.Dataset.from_tensor_slices(d_obs_train_raw)

# zip the x and y training data together and shuffle, batch etc.
train_dataset = tf.data.Dataset.zip((F_train,
        d_obs_train)).shuffle(500).repeat().batch(data_batch_size)

# Same for validation.
F_valid = tf.data.Dataset.from_tensor_slices(F_valid_raw)
d_obs_valid = tf.data.Dataset.from_tensor_slices(d_obs_valid_raw)
valid_dataset = tf.data.Dataset.zip((F_valid,
        d_obs_valid)).shuffle(500).repeat().batch(data_batch_size)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
        train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure
# via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
validation_init_op = iterator.make_initializer(valid_dataset)


# Model begins here (i.e., thats where data enters, before we only worked with
# cells).
def run_model(F, d):
    """ Run on one line of the forward and one data point.
    """
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

    prior_misfit = tf.subtract(d, tf.matmul(F, m_prior))

    m_posterior = tf.add(
            m_prior,
            tf.matmul(
                tf.matmul(pushforward_cov, inversion_operator),
                prior_misfit))
    prediction = tf.matmul(F, m_posterior)

    # Maximum likelyhood estimator of posterior mean, given values
    # of sigma and lambda. Obtained using the concentration formula.
    log_likelyhood = tf.add(
          log_det,
          tf.matmul(
              tf.transpose(prior_misfit),
              tf.matmul(inversion_operator, prior_misfit)))

    return (log_likelyhood, prediction, m_posterior)


log_likelyhood, prediction, _  = run_model(next_element[0], next_element[1])
loss = log_likelyhood
accuracy = tf.losses.mean_squared_error(prediction, next_element[1])

# Setup optimization
adam = tf.train.AdamOptimizer(learning_rate=1.0)
min_handle = adam.minimize(loss, var_list=[m0, sigma, length_scale])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)

    epochs = 200
    for i in range(epochs):
        print("Iteration {}".format(i))
        lkl, acc, _ = sess.run([loss, accuracy, min_handle])
        print("Epoch: {}, loss: {}, training accuracy: {}".format(i,
                str(lkl), str(acc)))
        print("m0: {}".format(sess.run(m0)))
        print("sigma: {}".format(sess.run(sigma)))
        print("lambda: {}".format(sess.run(length_scale)))

    valid_iters = 20
    # re-initialize the iterator, but this time with validation data
    sess.run(validation_init_op)
    avg_acc = 0.0
    for i in range(valid_iters):
        acc = sess.run([accuracy])
        avg_acc += acc[0]
    print("Average validation set accuracy over {} iterations is {:.10f}".format(
                    valid_iters, (avg_acc / valid_iters)))
