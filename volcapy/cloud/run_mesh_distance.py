# File: run_mesh_distance.py, Author: Cedric Travelletti, Date: 06.03.2019.
""" Try running the whole mesh euclidean distance in memory.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl

# GLOBALS
m0 = 2200.0
data_std = 0.1


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

# Only keep the first 50000 inversion cells, so the whole covariance matrix
# fits into memory.
inverseProblem.subset(30000)

distance_mesh = cl.compute_mesh_squared_euclidean_distance(
        inverseProblem.cells_coords[:, 0], inverseProblem.cells_coords[:, 0],
        inverseProblem.cells_coords[:, 1], inverseProblem.cells_coords[:, 1],
        inverseProblem.cells_coords[:, 2], inverseProblem.cells_coords[:, 2])

# Now torch in da place.
import torch
torch.set_num_threads(4)

distance_mesh = torch.as_tensor(distance_mesh)
F = torch.as_tensor(inverseProblem.forward)

m_prior = torch.mul(m0, torch.ones((inverseProblem.n_model, 1)))
data_cov = torch.mul(data_std**2, torch.eye(inverseProblem.n_data))

d_obs = torch.as_tensor(inverseProblem.data_values)


def posterior_squared_exponential(distance_mesh, inv_lambda_2, sigma_2):
    """ Squared exponential kernel. Builds the full covariance matrix from a
    squared distance mesh.

    Parameters
    ----------
    lambda_2: float
        Length scale parameter, in the form 1(2 * lambda^2).
    sigma_2: float
        (Square of) standard deviation.

    """
    pushforward_cov = (torch.mm(
            torch.mul(
            torch.exp(torch.mul(distance_mesh, - inv_lambda_2)),
            sigma_2),
            F.t()
            ))
    inversion_operator = torch.inverse(
            torch.add(
                    data_cov,
                    torch.mm(F, pushforward_cov)
                    ))
    m_posterior = torch.add(
            m_prior,
            torch.mm(
                    torch.mm(pushforward_cov, inversion_operator),
                    torch.sub(d_obs, torch.mm(F, m_prior)))
            )
    return m_posterior

# Matrix multiplication.
c = torch.mm(F, a)

# Elementwise exponential (underscore means in-place).
a = a.exp_()
