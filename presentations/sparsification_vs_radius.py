""" Study the number of non-zero entries in the covariance matrix as a function
of the threshold distance.

Note: The threshold distance is the distance between a couple of cells above
which we put the corresponding element in the covariance matrix to 0.

"""
from volcapy.inverse.flow import InverseProblem
import volcapy.grid.covariance_tools as cl
import numpy as np

from sklearn.neighbors import BallTree


# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)

tree = BallTree(inverseProblem.cells_coords)

def get_cells_ind_with_radius(cell, radius):
    """ Given a BallTree (based on an underlying list of cells), get indices of
    all the cells that are with radius of the given cell (actually, can be any
    point, not necessarily a cell of the BallTree).

    Parameters
    ----------
    cell: List[float, float, float]
        Coordinate of the point to consider.
    radius: float

    Returns
    -------
    List[int]
        Indices of the cells that are within the given radius.

    """
    return tree.query_radius(cell.reshape(1, -1), radius)[0]

def count_sparse_distance_mesh(threshold_dist):
    """ Does what is explained below, but we only COUNT the number of nonzero
    entries.

    Compute the n_model * n_model matrix of squared euclidean distances
    between all couples, ignoring couples with a distance greather than
    *threshold_dist*.

    Parameters
    ----------
    threshold_dist: float
        Only consider couples with distance less than this.

    Returns
    -------
    squared_dists: List[float]
        List of squared distances between couples of cells.
    inds: List[int]
        Gives the indices inside the full matrix, of the values in the
        preceding list.
        

    """
    count = 0

    # Loop over all cells.
    for cell_ind, cell in enumerate(inverseProblem.cells_coords):
        # Get neighbors.
        neighbor_inds = get_cells_ind_with_radius(cell, threshold_dist)
        count += len(neighbor_inds)

    return count
radiuses = []
counts = []
for radius in np.arange(50.0, 5000.0, 50.0):
    print(radius)
    count = count_sparse_distance_mesh(radius)
    print(count)

    radiuses.append(radius)
    counts.append(count)
