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
start_sigma0 = 190.0

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


def compute_CM_tilde(lambda0):
    """ Given lambda0, compute Cm * G.T, without the sigma0^2 prefactor.

    """
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
    return tot


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.sigma0 = torch.nn.Parameter(torch.tensor(start_sigma0).cuda())

        # Empty prior. Will be calculated using concentration formula.
        self.concentrated_m0 = torch.Tensor().to(device)

        self.F = F.to(device)
        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

        # Identity vector. Need for concentration.
        self.I = torch.ones((n_model, 1), dtype=torch.float32,
                        device=device)

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

        # We now concentrate the log-likelihood around m0.
        # Since the concetrated version of the prior mean is big, and since it
        # only plays a role when multiplied wiht the forward, we do not compute
        # it directly.
        self.concentrated_m0 = torch.mm(
            torch.inverse(
                torch.mm(
                    torch.mm(self.I.t(), self.F.t()),
                    torch.mm(
                        inversion_operator,
                        torch.mm(self.F, self.I)))),
            torch.mm(
                torch.mm(self.d_obs.t(), inversion_operator),
                torch.mm(self.F, self.I)))

        m_prior = self.concentrated_m0 * torch.ones((n_model, 1), device=device)
        prior_misfit = torch.sub(self.d_obs, torch.mm(self.F, m_prior))

        m_posterior = torch.add(
                m_prior,
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)


lambda0 = 200.0
tot = compute_CM_tilde(lambda0)

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

lambda0_start = 45.0
lambda0_stop = 5000.0
lambda0_step = 50.0

lambda0s = np.arange(lambda0_start, lambda0_stop + 0.1, lambda0_step)


print("Total number of lambda steps {}".format(len(m0s)))
nr_epochs = 100

for i, lambda0 in enumerate(lambda0s):
    print("Current lambda0 {}".format(lambda0))

    # Optimize.
    for epoch in range(nr_epochs):
        log_likelihood, m_posterior = model(tot)

        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        log_likelihood.backward(retain_graph=True)
        optimizer.step()
    
    # Compute prediction error.
    train_error = torch.sqrt(torch.mean(
        (d_obs.to(device) - torch.mm(F.to(device), m_posterior))**2))
    
    print("RMSE train error: {}".format(train_error))
    print("Log-likelihood: {}".format(log_likelihood.item()))
    print("Params: m0 {}, sigma0 {}.".format(
            model.concentrated_m0.item(), model.sigma0.item()))

    # Save data for each lambda.
    lls.append(log_likelihood.item())
    train_rmses.append(train_error.item())
    m0s.append(model.concentrated_m0.item())
    sigma0s.append(model.sigma0.item())

# Save results for plotting.
np.save("lambda0s_linesearch.npy", lambda0s)
np.save("lls_linesearch.npy", lls)
np.save("train_rmses_linesearch.npy", train_rmses)
np.save("m0s_linesearch.npy", m0s)
np.save("sigma0s_linesearch.npy", sigma0s)
