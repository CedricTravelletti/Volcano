""" Cressie Approach: Use Basis Functions centered at m different inducing
points.
"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

import torch
# torch.set_num_threads(3)


# GLOBALS
m0 = torch.tensor(2200.0, requires_grad=False).cuda()
data_std = torch.tensor(0.1, requires_grad=False).cuda()
sigma_epsilon = data_std
sigma_0 = torch.tensor(50.0, requires_grad=False).cuda()

n_basis = 1500
n_dims = torch.tensor(3, requires_grad=False).cuda()

# Initial length scales.
lambda0 = torch.tensor(2 * 800**2, requires_grad=False).cuda()
length_scales_init = lambda0 * torch.ones((n_basis, 1), dtype=torch.float32).cuda()

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/home/ec2-user/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Subset the data and build the train/test datasets.
F_test, d_obs_test = inverseProblem.subset_data(500)
n_model = torch.tensor(inverseProblem.n_model, requires_grad=False).cuda()

F_train = torch.as_tensor(inverseProblem.forward).cuda()
# Careful: we have to make a column vector here.
d_obs_train = torch.as_tensor(inverseProblem.data_values[:, None]).cuda()

data_cov = torch.mul(data_std**2, torch.eye(inverseProblem.n_data)).cuda()

# Put all points in an array.
# We transpose so the array is n_dim x n_cells.
coords = torch.as_tensor(inverseProblem.cells_coords)

# Randomly sample inducing points from the available cells.
inducing_points = torch.from_numpy(
        inverseProblem.cells_coords[np.random.choice(coords.shape[0], n_basis, replace=False)])

x = inducing_points.unsqueeze(1).expand(n_basis, n_model, n_dims)
y = coords.unsqueeze(0).expand(n_basis, n_model, n_dims)
dist = torch.pow(x - y, 2).sum(2).cuda()


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(m0).cuda()
        self.sigma_0 = torch.nn.Parameter(sigma_0).cuda()

        self.m_prior = torch.mul(self.m0, torch.ones((n_model, 1)).cuda()).cuda()
        self.data_cov = data_cov.cuda()

        # Initial lengthscales
        self.length_scales = torch.nn.Parameter(length_scales_init).cuda()

        self.PHI = torch.exp(- dist / self.length_scales).cuda()

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
        pushforward_cov = torch.mul(self.sigma_0**2,
                torch.mm(
                    self.PHI.t(),
                    torch.mm(self.PHI, F_train.t()))
                )
        print(pushforward_cov)
        inv_inversion_operator = torch.add(
                        self.data_cov,
                        torch.mm(F_train, pushforward_cov)
                        )
        inversion_operator = torch.inverse(inv_inversion_operator)

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = torch.logdet(inv_inversion_operator)
        print(log_det)

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

        # return log_likelyhood
        return log_likelyhood


myModel = SquaredExpModel()
myModel = myModel.cuda()

optimizer = torch.optim.Adam(myModel.parameters(), lr=5.0)
criterion = torch.nn.MSELoss().cuda()

for epoch in range(100):

    # Forward pass: Compute predicted y by passing
    # x to the model
    output = myModel().cuda()

    # Compute and print loss
    # loss = criterion(torch.mm(F_train, output), d_obs_train)
    loss = output

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(
            'epoch {}, loss {}, m0 {} length_scales {}, sigma {}'.format(
                    epoch, loss,
                    float(myModel.m0),
                    myModel.length_scales[0],
                    float(myModel.sigma_0)
                    ))
