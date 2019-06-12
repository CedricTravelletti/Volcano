# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Hyperparameter Optimization.

Lambda0 by brute force grid search, m0 by concentration, sigma0 by gradient
descent.

THIS VERSION EXPLICITLY RE-USED THE LAST SIGMA0 in order to make training
faster.

KERNEL
------
Simple exponential, no square.

ONLY TRAIN ON THE DATA SIDE, I.E.: DONT care about model side.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# Monitoring memory usage.
import os
import psutil
process = psutil.Process(os.getpid())

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
sigma0 = 20.0

###########
# IMPORTANT
###########
out_folder = "/idiap/temp/ctravelletti/out/data_side/"

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
# niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Test/Train split.
# Train-validation split.
nr_train = 450
F_test, d_obs_test = inverseProblem.subset_data(nr_train)

F_test = torch.as_tensor(F_test)
d_obs_test = torch.as_tensor(d_obs_test)

n_model = inverseProblem.n_model
n_data = inverseProblem.n_data

F = torch.as_tensor(inverseProblem.forward)
print("Size of model after regridding: {} cells.".format(n_model))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values[:, None])

cells_coords = torch.as_tensor(inverseProblem.cells_coords).detach().to(device)
del(inverseProblem)

# distance_mesh = torch.as_tensor(distance_mesh)
F = F.to(device)
data_cov = torch.mul(data_std**2, torch.eye(n_data))


def compute_K_d(lambda0):
    """ Compute the data-side kernel.

    """
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    inv_lambda2 = - 1 / (2 * lambda0**2)
    n_dims = 3
    tot = torch.Tensor().to(device)
    for i, x in enumerate(torch.chunk(cells_coords, chunks=150, dim=0)):
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
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.sigma0 = torch.nn.Parameter(torch.tensor(sigma0).cuda())

        # Empty prior. Will be calculated using concentration formula.
        self.concentrated_m0 = torch.Tensor().to(device)

        # Prior mean on the data side, stripped of the parameter.
        self.fwd_m0 = torch.mm(F, torch.ones((n_model, 1))).to(device)
        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

        # Identity vector. Need for concentration.
        self.I_d = torch.ones((n_data, 1), dtype=torch.float32,
                        device=device)

    def forward(self, tot):
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

        # Need to do it this way, otherwise rounding errors kill everything.
        log_det = - torch.logdet(inversion_operator)

        # We now concentrate the log-likelihood around m0.
        # Since the concetrated version of the prior mean is big, and since it
        # only plays a role when multiplied wiht the forward, we do not compute
        # it directly.
        self.concentrated_m0 = torch.mm(
            torch.inverse(
                torch.mm(
                    torch.mm(self.I_d.t(), inversion_operator),
                    self.I)),
            torch.mm(
                self.I_d.t(),
                torch.mm(inversion_operator, self.d_obs)))

        m_prior_d = self.concentrated_m0 * self.fwd_m0
        prior_misfit = torch.sub(self.d_obs, m_prior_d)

        weights = torch.mm(inversion_operator), prior_misfit)

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


        return (log_likelihood, m_posterior)

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

model = SquaredExpModel()
model = model.cuda()
# model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

lambda0_start = 60.0
lambda0_stop = 400.0
lambda0_step = 5.0
lambda0s = np.arange(lambda0_start, lambda0_stop + 0.1, lambda0_step)
n_lambda0s = len(lambda0s)
print("Number of lambda0s: {}".format(n_lambda0s))

n_epochs_short = 5000
n_epochs_long = 20000

lls = np.zeros((n_lambda0s), dtype=np.float32)
train_rmses = np.zeros((n_lambda0s), dtype=np.float32)
test_rmses = np.zeros((n_lambda0s), dtype=np.float32)
m0s = np.zeros((n_lambda0s), dtype=np.float32)
sigma0s = np.zeros((n_lambda0s), dtype=np.float32)


for i, lambda0 in enumerate(lambda0s):
    print("Current lambda0 {} , {} / {}".format(lambda0, i, n_lambda0s))
    # Correpsonding Cm tilde.
    K_d = compute_K_d(lambda0)

    # Perform the first training in full.
    # For the subsequent one, we can initialize sigma0 with the final value
    # from last training, since the optimum varies continuously in lambda0.
    # Hence, subsequent trainings can be shorter.
    if i > 0:
        n_epochs = n_epochs_short
    else: n_epochs = n_epochs_long

    # Optimize the tow remaining hyperparams.
    for epoch in range(n_epochs):
        # Forward pass: Compute predicted y by passing
        # x to the model
        log_likelihood, m_posterior = model(tot)

        # Compute train error.
        train_error = torch.sqrt(torch.mean(
            (d_obs.to(device) - torch.mm(F.to(device), m_posterior))**2))

        # Compute test error.
        test_error = torch.sqrt(torch.mean(
            (d_obs_test - torch.mm(F_test,
                    m_posterior.to(torch.device("cpu"))))**2))

        # Save data for each lambda.
        # Save only every 100 steps.
        if epoch % 100 == 0:
            print("Epoch {}".format(epoch))
            print("RMSE train error: {}".format(train_error.item()))
            print("RMSE test error: {}".format(test_error.item()))
            print("Log-likelihood: {}".format(log_likelihood.item()))
            print("Params: m0 {}, sigma0 {}.".format(
                    model.concentrated_m0.item(), model.sigma0.item()))

        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        log_likelihood.backward(retain_graph=True)
        optimizer.step()

    lls[i] = log_likelihood.item()
    train_rmses[i] = train_error.item()
    test_rmses[i] = test_error.item()
    m0s[i] = model.concentrated_m0.item()
    sigma0s[i] = model.sigma0.item()

    # Save every 4 lambdas.
    if i % 4 == 0:
        logger.info("Saving Results at lambda0 {} , {} / {}".format(lambda0, i, n_lambda0s))
        logger.info("Current memory usage: {} Gb".format(process.memory_info().rss / (1024**3)))

        np.save(os.path.join(out_folder, "log_likelihoods_train.npy"), lls)
        np.save(os.path.join(out_folder, "train_rmses_train.npy"), train_rmses)
        np.save(os.path.join(out_folder, "test_rmses_train.npy"), test_rmses)
        np.save(os.path.join(out_folder, "m0s_train.npy"), m0s)
        np.save(os.path.join(out_folder, "sigma0s_train.npy"), sigma0s)
        np.save(os.path.join(out_folder, "lambda0s_train.npy"), lambda0s)

logger.info("Finished. Saving results")
np.save(os.path.join(out_folder, "log_likelihoods_train.npy"), lls)
np.save(os.path.join(out_folder, "train_rmses_train.npy"), train_rmses)
np.save(os.path.join(out_folder, "test_rmses_train.npy"), test_rmses)
np.save(os.path.join(out_folder, "m0s_train.npy"), m0s)
np.save(os.path.join(out_folder, "sigma0s_train.npy"), sigma0s)
np.save(os.path.join(out_folder, "lambda0s_train.npy"), lambda0s)