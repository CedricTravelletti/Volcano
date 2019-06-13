# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

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
torch.set_num_threads(1)
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
F_cpu = torch.as_tensor(inverseProblem.forward).detach()
F = F_cpu.to(device)

# Careful: we have to make a column vector here.
data_std = 0.1
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])
data_cov = torch.mul(data_std**2, torch.eye(n_data))

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach().to(device)
del(inverseProblem)
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# ----------------------------------------------------------------------------#
#     HYPERPARAMETERS
# ----------------------------------------------------------------------------#
sigma0_init = 700.0
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
    for i, x in enumerate(torch.chunk(cells_coords, chunks=120, dim=0)):
        print(i)
        # Empty cache every so often. Otherwise we get out of memory errors.
        if i % 10 == 0:
            pass
            # torch.cuda.empty_cache()

        tot = torch.cat((
                tot,
                torch.matmul(torch.exp(inv_lambda2
                    * torch.pow(
                        x.unsqueeze(1).expand(x.shape[0], n_model, n_dims) -
                        cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
                        , 2).sum(2))
                    , F.t())))
    print("Computing final.")
    final = torch.mm(F, tot)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Computed final.")

    # Send back to CPU.
    return final.cpu()


class SquaredExpModel(torch.nn.Module):
    def __init__(self, F):
        super(SquaredExpModel, self).__init__()

        # Store the sigma0 after optimization, since can be used as starting
        # point for next optim.
        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0_init))

        # Prior mean (vector) on the data side.
        self.mu0_d_stripped = torch.mm(F, torch.ones((n_model, 1)))

        self.d_obs = d_obs
        self.data_cov = data_cov

        # Identity vector. Need for concentration.
        self.I_d = torch.ones((n_data, 1), dtype=torch.float32)

    def forward(self, K_d, sigma0, m0=0.1, concentrate=False):
        logger.debug("GPU used before forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inv_inversion_operator = torch.add(
                        self.data_cov,
                        sigma0**2 * K_d)

        logger.debug("GPU used after computing inv_inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        # Compute inversion operator and store once and for all.
        self.inversion_operator = torch.inverse(inv_inversion_operator)

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(self.inversion_operator)

        logger.debug("GPU used after computing inversion_operator: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))
        del inv_inversion_operator

        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            m0 = torch.mm(
                torch.inverse(
                    torch.mm(
                        torch.mm(self.mu0_d_stripped.t(), self.inversion_operator),
                        self.mu0_d_stripped)),
                torch.mm(
                    self.mu0_d_stripped.t(),
                    torch.mm(self.inversion_operator, self.d_obs)))

        self.mu0_d = m0 * self.mu0_d_stripped
        # Store m0 in case we want to print it later.
        self.m0 = m0
        self. prior_misfit = torch.sub(self.d_obs, self.mu0_d)
        weights = torch.mm(self.inversion_operator, self.prior_misfit)

        m_posterior_d = torch.add(
                self.mu0_d,
                torch.mm(sigma0**2 * K_d, weights))

        # Maximum likelihood estimator of posterior mean, given values
        # of sigma and lambda. Obtained using the concentration formula.
        log_likelihood = torch.add(
              log_det,
              torch.mm(
                      self.prior_misfit.t(),
                      torch.mm(self.inversion_operator, self.prior_misfit)))

        logger.debug("GPU at end of forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        return (log_likelihood, m_posterior_d)

    def optimize(self, K_d, n_epochs):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        Parameters
        ----------
        K_d: 2D Tensor
            (stripped) Covariance matrix in data space.
        sigma0_init: float
            Starting value for gradient descent.
        n_epochs: int
            Number of training epochs.

        """
        # Send everything to GPU first.
        self.sigma0 = self.sigma0.to(device)
        self.mu0_d_stripped = self.mu0_d_stripped.to(device)
        self.d_obs = self.d_obs.to(device)
        self.data_cov = self.data_cov.to(device)
        self.I_d = self.I_d.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.5)
        for epoch in range(n_epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            log_likelihood, m_posterior_d = model(K_d, self.sigma0, concentrate=True)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            log_likelihood.backward(retain_graph=True)
            optimizer.step()

            # Compute train error.
            train_error = torch.sqrt(torch.mean(
                (self.d_obs - m_posterior_d)**2))
            
            print("Log-likelihood: {}".format(log_likelihood.item()))
            print("RMSE train error: {}".format(train_error.item()))


    def loo_predict(self, loo_ind):
        """ Leave one out krigging prediction.

        Take the trained hyperparameters. Remove one point from the
        training set, krig/condition on the remaining point and predict
        the left out point.

        WARNING: Should have run the forward pass of the model once before,
        otherwise some of the operators we need (inversion operator) won't have
        been computed. Also should re-run the forward pass when updating
        hyperparameters.

        Parameters
        ----------
        loo_ind: int
            Index (in the training set) of the data point to leave out.

        Returns
        -------
        float
            Prediction at left out data point.

        """
        # Index of the not removed data points.
        in_inds = list(range(len(self.d_obs)))
        in_inds.remove(loo_ind)

        # Note that for the dot product, we should have one-dimensional
        # vectors, hence the strange indexing with the zero.
        loo_pred = (self.mu0_d[loo_ind] -
                1/self.inversion_operator[loo_ind, loo_ind] *
                torch.dot(
                    self.inversion_operator[loo_ind, in_inds],
                    self.prior_misfit[in_inds, 0]))

        return loo_pred

    def loo_error(self):
        """ Leave one out cross validation RMSE.

        Take the trained hyperparameters. Remove one point from the
        training set, krig/condition on the remaining point and predict
        the left out point.
        Compute the squared error, repeat for all data points (leaving one out
        at a time) and average.

        WARNING: Should have run the forward pass of the model once before,
        otherwise some of the operators we need (inversion operator) won't have
        been computed. Also should re-run the forward pass when updating
        hyperparameters.

        Returns
        -------
        float
            RMSE cross-validation error.

        """
        tot_error = 0
        for loo_ind in range(len(self.d_obs)):
            loo_prediction = self.loo_predict(loo_ind)
            tot_error += (self.d_obs[loo_ind].item() - loo_prediction**2)

        return np.sqrt((tot_error / len(self.d_obs)))

K_d = compute_K_d(lambda0, F)
model = SquaredExpModel(F_cpu)

# Predict
log_likelihood, m_posterior_d = model(K_d, sigma0=500.0, m0=2200.0)
print("Log-likelihood: {}".format(log_likelihood.item()))

# Predict with concentration.
log_likelihood, m_posterior_d = model(K_d, sigma0=250.0, concentrate=True)
print("Log-likelihood: {}".format(log_likelihood.item()))
print("m0: {}".format(model.m0))

# Compute train error.
train_error = torch.sqrt(torch.mean(
    (model.d_obs - m_posterior_d)**2))
print("RMSE train error: {}".format(train_error.item()))


from timeit import default_timer as timer
model = model.cuda()
start = timer()
model.optimize(K_d.to(device), 1000)
end = timer()
print(end - start)

# print(model.loo_predict(10))
