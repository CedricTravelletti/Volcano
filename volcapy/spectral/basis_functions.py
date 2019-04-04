""" Cressie Approach: Use Basis Functions centered at m different inducing
points.
"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

import torch
torch.set_num_threads(3)


# GLOBALS
m0 = 2200.0
data_std = 0.1
sigma_epsilon = data_std
sigma_0 = 1.0

n_basis = 1000
n_dims = 3

# Initial length scales.
lambda0 = 300**2
length_scales_init = lambda0 * torch.ones((n_basis, 1), dtype=torch.float32)

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Subset the data and build the train/test datasets.
F_test, d_obs_test = inverseProblem.subset_data(500)
n_model = inverseProblem.n_model

F_train = torch.as_tensor(inverseProblem.forward)
# Careful: we have to make a column vector here.
d_obs_train = torch.as_tensor(inverseProblem.data_values[:, None])

data_cov = torch.mul(data_std**2, torch.eye(inverseProblem.n_data))

# Put all points in an array.
# We transpose so the array is n_dim x n_cells.
coords = torch.as_tensor(inverseProblem.cells_coords)

# Randomly sample inducing points from the available cells.
inducing_points = torch.from_numpy(
        inverseProblem.cells_coords[np.random.choice(coords.shape[0], n_basis, replace=False)])

x = inducing_points.unsqueeze(1).expand(n_basis, n_model, n_dims)
y = coords.unsqueeze(0).expand(n_basis, n_model, n_dims)
dist = torch.pow(x - y, 2).sum(2)


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(torch.tensor(m0))
        self.sigma_0 = torch.nn.Parameter(torch.tensor(sigma_0))

        self.m_prior = torch.mul(self.m0, torch.ones((inverseProblem.n_model, 1)))
        self.data_cov = data_cov

        # Initial lengthscales
        self.length_scales = torch.nn.Parameter(length_scales_init)

        self.PHI = torch.exp(- dist / self.length_scales)

    def forward(self):
        """ Squared exponential kernel. Builds the full covariance matrix from a
        squared distance mesh.

        Parameters
        ----------
        lambda_2: float
            Length scale parameter, in the form 1(2 * lambda^2).
        sigma_2: float
            (Square of) standard deviation.

        """
        prior_misfit = torch.sub(d_obs_train, torch.mm(F_train, self.m_prior))

        # Compute C_M GT.
        pushforward_cov = torch.mul(sigma_0**2,
                torch.mm(
                    self.PHI.t(),
                    torch.mm(self.PHI, F_train.t()))
                )
        inv_inversion_operator = torch.add(
                        self.data_cov,
                        torch.mm(F_train, pushforward_cov)
                        )
        inversion_operator = torch.inverse(inv_inversion_operator)

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        prior_misfit = torch.sub(d_obs_train, torch.mm(F_train, self.m_prior))

        m_posterior = torch.add(
                self.m_prior,
                torch.mm(
                        torch.mm(pushforward_cov, inversion_operator),
                        prior_misfit)
                )
        # Maximum likelyhood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelyhood = torch.add(
              log_det,
              torch.mm(
                      prior_misfit.t(),
                      torch.mm(inversion_operator, prior_misfit)))

        return log_likelyhood


myModel = SquaredExpModel()
optimizer = torch.optim.Adam(myModel.parameters(), lr=1.3)
criterion = torch.nn.MSELoss()

for epoch in range(100):

    # Forward pass: Compute predicted y by passing
    # x to the model
    log_likelyhood = myModel()

    # Compute and print loss
    loss = log_likelyhood

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(
            'epoch {}, loss {}, m0 {} length_scales {}, sigma {}'.format(
                    epoch, loss.data,
                    float(myModel.m0),
                    myModel.length_scales[0],
                    float(myModel.sigma_0)
                    ))
