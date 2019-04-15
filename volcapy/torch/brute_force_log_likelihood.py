# File: brute_force_log_likelihood.py, Author: Cedric Travelletti, Date: 15.04.2019.
""" Only do forward pass. Brute force optimization by grid search.

This is an attempt at saving what can be saved, in the aftermath of the April
11 discovery that everything was wrong due to test-train split screw up.

Lets hope it works.

THIS ONE DOES NOT USE THE CONCENTRATED VERSION, BUT RUNS GRADIENT DESCENT ON m0
AND sigma0 instead.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# Set up logging.
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Now torch in da place.
import torch
torch.set_num_threads(4)
# Choose between CPU and GPU.
device = torch.device('cuda:0')
# device = torch.device('cpu')

# GLOBALS
data_std = 0.1
sigma_0 = 88.95
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
d_obs = torch.as_tensor(inverseProblem.data_values)

"""
cl.save_mesh_squared_euclidean_distance(
        "squared_dist_mesh.npy",
        cells_coords[:, 0], cells_coords[:, 0],
        cells_coords[:, 1], cells_coords[:, 1],
        cells_coords[:, 2], cells_coords[:, 2])
"""
del(inverseProblem)

dist_mesh = np.memmap("/scratch/ctravelletti/squared_dist_mesh.npy", dtype='float32', mode='r',
                shape=(n_model, n_model))

# distance_mesh = torch.as_tensor(distance_mesh)
F = F.to(device)
data_cov = torch.mul(data_std**2, torch.eye(n_data))

# Distance mesh is horribly expansive, to use half-precision.
# distance_mesh = distance_mesh.to(torch.device("cpu"))


def per_chunk_pushforward_cov(chunk, lambda0):
    """ Given a chunk of the squared distance mesh, compute its associated part
    of the pushforward covariance.

    """
    inv_lambda2 = - 1 / (2 * lambda0**2)
    return torch.matmul(torch.exp(torch.mul(inv_lambda2, chunk)), F.t())

def compute_pushforward_cov(lambda0, dist_mesh):
    pushforward_cov = torch.Tensor().to(device)
    # Produce the chunks and iterate in chunks;
    chunks = mat.chunk_range(dist_mesh.shape[0], chunk_size=6000)
    for row_begin, row_end in chunk:
        print(row_begin)
        chunk = torch.as_tensor(dist_mesh[row_begin:row_end + 1, :]).to(device)
        torch.cat(pushforward_cov,
                per_chunk_pushforward_cov(chunk, lambda0))
    return pushforward_cov


class SquaredExpModel(torch.nn.Module):
    def __init__(self):
        super(SquaredExpModel, self).__init__()

        self.m0 = torch.nn.Parameter(torch.tensor(m0).cuda())
        self.sigma_0 = torch.nn.Parameter(torch.tensor(sigma_0).cuda())

        self.m_prior = torch.mul(self.m0, torch.ones((n_model,
                1), device=device)).cuda()

        self.F = F.to(device)
        self.d_obs = d_obs.to(device)
        self.data_cov = data_cov.to(device)

    def forward(self, pushforward_cov):
        # torch.cuda.empty_cache()
        logger.debug("GPU used before forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        inv_inversion_operator = torch.add(
                        self.data_cov,
                        torch.mul(
                            self.sigma_0.pow(2),
                            torch.mm(self.F, pushforward_cov))
                        )
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

        print(log_likelyhood.shape)
        logger.debug("GPU at end of forward pass: {} Gb.".format(
                torch.cuda.memory_allocated(device)/1e9))

        return (log_likelyhood, m_posterior)


model = SquaredExpModel()
model = torch.nn.DataParallel(model).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2.0)

losses = []
train_rmses = []
m0s = []
sigma0s = []

# Iterate over candidate lengthscales.
lambda0s = np.linspace(start=45.0, stop=5000.0, num=100)

for i, lambda0 in enumerate(lambda0s):
    lambda0 = torch.tensor(lambda0).to(device)

    # Compute the corresponding pushforward cov.
    pushforward_cov = compute_pushforward_cov(lambda0, dist_mesh)

    # Optimize the tow remaining hyperparams.
    for epoch in range(100000):
    
        # Forward pass: Compute predicted y by passing
        # x to the model
        log_likeyhood, m_posterior = model(pushforward_cov)
    
        # Compute and print loss
        loss = log_likelyhood
    
        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()
    
        # Compute prediction error.
        criterion = torch.nn.MSELoss()
        train_error = torch.sqrt(criterion(
                torch.mm(F.to(device), m_posterior), d_obs.to(device)))
    
        print("RMSE train error: {}".format(train_error))
        print("Log-likelihood: {}".format(loss))

    # Save data for each lambda.
    losses.append(loss.data.cpu().numpy()[0])
    train_rmses.append(train_error.data.cpu())
    m0s.append(model.m0.data.cpu())
    sigma0s.append(model.sigma_0.data.cpu())

print("Saving Results ...")
np.save("losses.npy", np.array(losses))
np.save("train_rmses.npy", np.array(train_rmses))
np.save("m0s.npy", np.array(m0s))
np.save("sigma0s.npy", np.array(sigma0s))
np.save("lambda0s.npy", np.array(lambda0s))
