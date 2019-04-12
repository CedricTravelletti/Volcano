# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Only do forward pass. Brute force optimization by grid search.

This is an attempt at saving what can be saved, in the aftermath of the April
11 discovery that everything was wrong due to test-train split screw up.

Lets hope it works.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
from volcapy.grid.regridding import irregular_regrid_single_step, regrid_forward
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

# Regrid the problem at lower resolution.
coarse_cells_coords, coarse_to_fine_inds = irregular_regrid_single_step(
        inverseProblem.cells_coords, 100.0)

# Save a regridded version before splitting
F_coarse_tot = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
np.save("F_coarse_tot.npy", F_coarse_tot)

n_data = inverseProblem.n_data

new_F = regrid_forward(inverseProblem.forward, coarse_to_fine_inds)
n_model = len(coarse_to_fine_inds)
del(coarse_to_fine_inds)
print("Size of model after regridding: {} cells.".format(n_model))

# Careful: we have to make a column vector here.
d_obs = torch.as_tensor(inverseProblem.data_values)

distance_mesh = cl.compute_mesh_squared_euclidean_distance(
        coarse_cells_coords[:, 0], coarse_cells_coords[:, 0],
        coarse_cells_coords[:, 1], coarse_cells_coords[:, 1],
        coarse_cells_coords[:, 2], coarse_cells_coords[:, 2])
del(coarse_cells_coords)
del(inverseProblem)

distance_mesh = torch.as_tensor(distance_mesh)
F = torch.as_tensor(new_F)
data_cov = torch.mul(data_std**2, torch.eye(n_data))

# Distance mesh is horribly expansive, to use half-precision.
distance_mesh = distance_mesh.to(torch.device("cpu"))

lambda0 = torch.tensor(lambda0, requires_grad=True)
inv_lambda2 = - 1 / (2 * lambda0**2)
a = torch.stack(
    [torch.matmul(torch.exp(torch.mul(inv_lambda2, x)), F.t()) for i, x in enumerate(torch.unbind(distance_mesh, dim=0))],
    dim=0)

# Try to do it in chunks.
b = torch.cat(
    [torch.matmul(torch.exp(torch.mul(inv_lambda2, x)), F.t()) for i, x in enumerate(torch.chunk(distance_mesh, chunks=20, dim=0))],
    dim=0)




def per_line_operation(line):
    """ Operation to perform on each line.
    That is, given a line of the distance mesh,
    return the corresponding line of the matrix exp(-dist^2 / 2 lambda^2).

    """
    out = tf.matmul(torch.exp(torch.mul(inv_lambda2, line)), F.t())
    return out

# Compute C_M GT.
pushforward_cov = tf.squeeze(
        tf.map_fn(per_line_operation, coords_subset))
