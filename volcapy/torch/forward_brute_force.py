# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Only do forward pass. Brute force optimization by grid search.

This is an attempt at saving what can be saved, in the aftermath of the April
11 discovery that everything was wrong due to test-train split screw up.

Lets hope it works.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

# Now torch in da place.
import torch
# Choose between CPU and GPU.
# device = torch.device('cuda:0')
device = torch.device('cpu')

# GLOBALS
data_std = 0.1
lambda0 = 164.17
sigma_0 = 88.95

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

n_data = inverseProblem.n_data
F = torch.as_tensor(inverseProblem.forward)
n_model = F.shape[1]
print("Size of model after regridding: {} cells.".format(n_model))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values)

cells_coords = inverseProblem.cells_coords
distance_mesh = cl.compute_mesh_squared_euclidean_distance(
        cells_coords[:, 0], cells_coords[:, 0],
        cells_coords[:, 1], cells_coords[:, 1],
        cells_coords[:, 2], cells_coords[:, 2])
del(inverseProblem)

distance_mesh = torch.as_tensor(distance_mesh)
F = torch.as_tensor(F)
data_cov = torch.mul(data_std**2, torch.eye(n_data))

# Distance mesh is horribly expansive, to use half-precision.
distance_mesh = distance_mesh.to(torch.device("cpu"))

lambda0 = torch.tensor(lambda0, requires_grad=True)
inv_lambda2 = - 1 / (2 * lambda0**2)
a = torch.stack(
    [torch.matmul(torch.exp(torch.mul(inv_lambda2, x)), F.t()) for i, x in enumerate(torch.unbind(distance_mesh, dim=0))],
    dim=0)

# Try to do it in chunks.
cells_coords = torch.as_tensor(cells_coords)
cells_coords = cells_coords[:100000, :]
F = F[:, :100000]
n_model = 100000

n_dims = 3
b = torch.cat(
    [torch.matmul(torch.exp(torch.mul(inv_lambda2,
            torch.pow(
                    inducing_points.unsqueeze(1).expand(inducing_points.shape[0], n_model, n_dims) - 
                    cells_coords.unsqueeze(0).expand(inducing_points.shape[0], n_model, n_dims)
                    , 2).sum(2)))
            , F.t()) for i, inducing_points in enumerate(torch.chunk(cells_coords, chunks=20, dim=0))],
    dim=0)

c = torch.cat(
    [torch.matmul(torch.exp(torch.mul(inv_lambda2, x)), F.t()) for i, x in enumerate(torch.chunk(distance_mesh, chunks=20, dim=0))],
    dim=0)
