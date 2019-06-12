# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# Now torch in da place.
import torch
torch.set_num_threads(4)
# Choose between CPU and GPU.
device = torch.device('cuda:0')


# ----------------------------------------------------------------------------#
#      LOAD NIKLAS DATA
# ----------------------------------------------------------------------------#
# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)
n_model = inverseProblem.n_model
n_data = inverseProblem.n_data
F = torch.as_tensor(inverseProblem.forward).to(device)

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
data_cov = torch.mul(data_std**2, torch.eye(n_data))

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach().to(device)
del(inverseProblem)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
data_std = 0.1
sigma0 = 700.0
m0 = 2000.0
lambda0 = 200.0
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#


def compute_K_d(lambda0, F):
    """ Compute the data-side covariance matrix.

    The data-side covariance matrix is just FKF^T, where K is the model
    covariance matrix.

    """
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    inv_lambda2 = - 1 / (2 * lambda0**2)
    n_dims = 3

    # Array to hold the results. We will compute line by line and concatenate.
    tot = torch.Tensor().to(device)

    # Compute K * F^T chunk by chunk.
    for i, x in enumerate(torch.chunk(cells_coords, chunks=150, dim=0)):
        # Empty cache every so often. Otherwise we get out of memory errors.
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

    return torch.mm(F, tot)


class SquaredExpModel(torch.nn.Module):
    def __init__(self, m0, sigma0, K_d, F):
        super(SquaredExpModel, self).__init__()

        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0).cuda())

        # Prior mean (vector) on the data side.
        self.mu0_d = torch.mm(F, torch.ones((n_model, 1), device=device))

        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

        # Identity vector. Need for concentration.
        self.I_d = torch.ones((n_data, 1), dtype=torch.float32,
                        device=device)

    def forward(self, K_d):
        # torch.cuda.empty_cache()
        logger.debug("GPU used before forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inv_inversion_operator = torch.add(
                        self.data_cov,
                        self.sigma0**2 * K_d)
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inv_inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inversion_operator = torch.inverse(inv_inversion_operator)
        torch.cuda.empty_cache()

        logger.debug("GPU used after computing inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))
        del inv_inversion_operator

        prior_misfit = torch.sub(self.d_obs, mu0_d)
        weights = torch.mm(inversion_operator, prior_misfit)

        m_posterior_d = torch.add(
                m_prior_d,
                torch.mm(self.sigma0**2 * K_d, weights))

        # Maximum likelihood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelihood = torch.add(
              log_det,
              torch.mm(
                      prior_misfit.t(),
                      torch.mm(inversion_operator, prior_misfit)))

        logger.debug("GPU at end of forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))


        return (log_likelihood, m_posterior_d)

    def loo_predict(self, loo_index):
        """ Leave one out krigging prediction.

        Take the trained hyperparameters. Remove one point from the
        training set, krig/condition on the remaining point and predict
        the left out point.

        Parameters
        ----------
        loo_index: int
            Index (in the training set) of the data point to leave out.

        """

K_d = compute_K_d(lambda0, F)
model = SquaredExpModel(K_d, F)
model = model.cuda()

# Correpsonding Cm tilde.

log_likelihood, m_posterior_d = model(K_d)

# Compute train error.
train_error = torch.sqrt(torch.mean(
    (d_obs.to(device) - m_posterior)**2))

