# File: run_mesh_distance.py, Author: Cedric Travelletti, Date: 06.03.2019.
""" Run maximum likelyhood parameter estimation. Use concentrated version to avoid
optimizing the prio mean.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

from sklearn.neighbors import BallTree


# GLOBALS
m0 = 2200.0
data_std = 0.1

length_scale = 100.0
sigma = 20.0


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

tree = BallTree(inverseProblem.cells_coords)

def get_cells_ind_with_radius(cell, radius):
    return tree.query_radius(cell.reshape(1, -1), radius)[0]

def compute_cov_within_radius(cell, radius):
    neighbor_inds = get_cells_ind_with_radius(cell, radius)
    for neighbor_ind in neighbor_inds:
        cov = np.exp(
                - np.linalg.norm(
                        cell - inverseProblem.cells_coords[neighbor_ind, :]))

for i in range(inverseProblem.cells_coords.shape[0]):
    print(i)
    compute_cov_within_radius(inverseProblem.cells_coords[i, :], 250.0)
