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

def compute_sparse_distance_mesh(radius):
    # Lists to store the computed covariances and indices in the big sparse
    # matrix.
    covariances = []
    inds = []

    # Loop over all cells.
    for cell_ind, cell in enumerate(inverseProblem.cells_coords):
        print(cell_ind)
        # Get neighbors.
        neighbor_inds = get_cells_ind_with_radius(cell, radius)

        # Loop over all neighboring cells.
        for neighbor_ind in neighbor_inds:
            covariances.append(
                np.linalg.norm(
                    cell - inverseProblem.cells_coords[neighbor_ind, :]))
            inds.append((cell_ind, neighbor_ind))
    return (covariances, inds)
