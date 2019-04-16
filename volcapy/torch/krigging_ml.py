# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Only do forward pass. Brute force optimization by grid search.

This is an attempt at saving what can be saved, in the aftermath of the April
11 discovery that everything was wrong due to test-train split screw up.

Lets hope it works.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# Set up logging.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now torch in da place.
import torch
torch.set_num_threads(4)
# Choose between CPU and GPU.
device = torch.device('cuda:0')
# device = torch.device('cpu')

# GLOBALS
data_std = 0.1
lambda0 = 200.0
sigma0 = 88.95
m0 = 2200.0

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

n_data = inverseProblem.n_data
F = torch.as_tensor(inverseProblem.forward)
n_model = F.shape[1]
print("Size of model after regridding: {} cells.".format(n_model))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach().to(device)
del(inverseProblem)

# distance_mesh = torch.as_tensor(distance_mesh)
F = F.to(device)
data_cov = torch.mul(data_std**2, torch.eye(n_data))

# Distance mesh is horribly expansive, to use half-precision.
# distance_mesh = distance_mesh.to(torch.device("cpu"))

lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
inv_lambda2 = - 1 / (2 * lambda0**2)

n_dims = 3
tot = torch.Tensor().to(device)
for i, x in enumerate(torch.chunk(cells_coords, chunks=150, dim=0)):
    print(i)
    if i % 80 == 0:
        torch.cuda.empty_cache()
    tot = torch.cat((
            tot,
            torch.matmul(torch.exp(inv_lambda2
                * torch.pow(
                    x.unsqueeze(1).expand(x.shape[0], n_model, n_dims) - 
                    cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
                    , 2).sum(2))
                , F.t())))


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(torch.tensor(m0).cuda())
        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0).cuda())

        self.m_prior = torch.mul(self.m0, torch.ones((n_model,
                1), device=device)).cuda()

        self.F = F.to(device)
        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

    def forward(self, tot):
        # torch.cuda.empty_cache()
        logger.debug("GPU used before forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inv_inversion_operator = torch.add(
                        self.data_cov,
                        self.sigma0**2 * torch.mm(self.F, tot))
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inv_inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inversion_operator = torch.inverse(inv_inversion_operator)
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))
        del inv_inversion_operator

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        prior_misfit = torch.sub(self.d_obs, torch.mm(self.F, self.m_prior))

        m_posterior = torch.add(
                self.m_prior,
                torch.mm(
                        torch.mm(self.sigma0**2 * tot, inversion_operator),
                        prior_misfit)
                )
        # Maximum likelihood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelihood = torch.add(
              log_det,
              torch.mm(
                      prior_misfit.t(),
                      torch.mm(inversion_operator, prior_misfit)))

        logger.debug("GPU at end of forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        return (log_likelihood, m_posterior)


model = SquaredExpModel()
model = model.cuda()
# model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# Forward pass: Compute predicted y by passing
# x to the model
log_likelihood, m_posterior = model(tot)

train_error = torch.sqrt(torch.mean(
    (d_obs.to(device) - torch.mm(F.to(device), m_posterior))**2))
print("RMSE train error: {}".format(train_error.item()))
print("Log-likelihood: {}".format(log_likelihood.item()))

lls = []
train_rmses = []
m0s = []
sigma0s = []


# Optimize the tow remaining hyperparams.
for epoch in range(100000):
    # Forward pass: Compute predicted y by passing
    # x to the model
    log_likelihood, m_posterior = model(tot)
    
    # Compute prediction error.
    train_error = torch.sqrt(torch.mean(
        (d_obs.to(device) - torch.mm(F.to(device), m_posterior))**2))
    
    print("RMSE train error: {}".format(train_error))
    print("Log-likelihood: {}".format(log_likelihood.item()))
    print("Params: m0 {}, sigma0 {}.".format(
            model.m0.item(), model.sigma0.item()))

    # Save data for each lambda.
    lls.append(log_likelihood.item())
    train_rmses.append(train_error.item())
    m0s.append(model.m0.item())
    sigma0s.append(model.sigma0.item())

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    log_likelihood.backward(retain_graph=True)
    optimizer.step()


print("Saving Results ...")
np.save("log_likelihoods.npy", np.array(lls))
np.save("train_rmses.npy", np.array(train_rmses))
np.save("m0s.npy", np.array(m0s))
np.save("sigma0s.npy", np.array(sigma0s))
np.save("lambda0s.npy", np.array(lambda0s))
