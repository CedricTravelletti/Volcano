# File: sparse_mesh.py, Author: Cedric Travelletti, Date: 11.03.2019.
""" Compactly supported Wendland Kernel.

"""
from volcapy.inverse.flow import InverseProblem
from volcapy.grid.sparsifier import Sparsifier
import numpy as np
import torch


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
sparsifier = Sparsifier(inverseProblem.cells_coords)
radius = 50.0

# Build the distance matrix as a sparse tensor.
dist_mesh_vals, dist_mesh_inds = sparsifier.compute_sparse_distance_mesh(radius)
dist_mesh_vals = torch.from_numpy(dist_mesh_vals)
dist_mesh_inds = torch.from_numpy(dist_mesh_inds)
dist_mesh_shape = np.array([n_model, n_model], dtype=np.int64)

# Initialization variables.
m0 = 2184.0
data_std = 0.1
length_scale = 50.0
sigma = 140.0

min_len = 0.0
max_len = 100.0



# Setup datasets.
from torch.utils.data import TensorDataset, DataLoader
batch_size = 20
train_ds = TensorDataset(
    torch.from_numpy(F_train_raw),
    torch.from_numpy(d_obs_train_raw))
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Same for validation.
valid_ds = TensorDataset(
    torch.from_numpy(F_valid_raw),
    torch.from_numpy(d_obs_valid_raw))
valid_dl = DataLoader(valid_ds, batch_size, shuffle=True)

# Model begins here (i.e., thats where data enters, before we only worked with
# cells).
class SquaredExpModel(torch.nn.Module):
    def __init__(self, batch_size):
        super(SquaredExpModel, self).__init__()
        self.m0 = torch.nn.Parameter(torch.tensor(m0))
        self.sigma = torch.nn.Parameter(torch.tensor(sigma))
        self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
        self.clipped_length_scale = torch.clamp(
            self.length_scale, min=min_len, max=max_len)

        self.m_prior = torch.mul(
            self.m0,
            torch.ones((inverseProblem.n_model, 1)))


        # Compute the values of the non-zero elements in the covariance matrix.
        # We operate on the list of non-zero values, not on the matrix.
        # This is easier, since taking the exp of zero values would unzero them.
        covariance_mat_vals = torch.mul(
            self.sigma.pow(2),
            torch.mul(
                torch.nn.functional.relu((1 - torch.div(dist_mesh_vals, 2 *
                        self.clipped_length_scale))).pow(4),
                (1 + torch.div(2 * dist_mesh_vals, self.clipped_length_scale))))

        # Now put this in a sparse matrix.
        self.covariance_mat_sparse = torch.sparse.FloatTensor(
                dist_mesh_inds.t(), covariance_mat_vals,
                torch.Size(dist_mesh_shape))

    def forward(self, F_rows, d_rows):
        """ Run on a batch of lines of the forward and a batch of data points
        """
        self.data_cov = torch.mul(
            data_std**2,
            torch.eye(d_rows.shape[0]))

        pushforward_cov = torch.sparse.mm(
                        self.covariance_mat_sparse,
                        F_rows.t())
        inv_inversion_operator = torch.add(
                self.data_cov,
                torch.mm(F_rows, pushforward_cov))
        inversion_operator = torch.inverse(inv_inversion_operator)

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        prior_misfit = torch.sub(d_rows, torch.mm(F_rows, self.m_prior))

        m_posterior = torch.add(
                self.m_prior,
                torch.mm(
                    torch.mm(pushforward_cov, inversion_operator),
                    prior_misfit))
        prediction = torch.mm(F_rows, m_posterior)

        # Maximum likelyhood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelyhood = torch.add(
            log_det,
            torch.mm(
                prior_misfit.t(),
                torch.mm(inversion_operator, prior_misfit)))

        return (log_likelyhood, prediction, m_posterior)

model = SquaredExpModel(data_batch_size)

# Setup optimization
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

epochs = 200
def fit(num_epochs, model, opt):
    for epoch in range(num_epochs):
        print("Iteration {}".format(epoch))
        # Train with batches of data.
        for xb, yb in train_dl:
            # 1. Generate predictions
            tmp = model(xb, yb)
            pred = tmp[1]

            # 2. Compute loss
            loss = tmp[0]

            # 3. Compute gradients
            # loss.backward()

            # 4. Update params using gradients.
            # opt.step()

            # 5. Reset gradients to zero.
            # opt.zero_grad()

        print("Epoch: {}, loss: {}, training accuracy: ".format(epoch,
                str(loss.item())))
