# File: sparsifier.py, Author: Cedric Travelletti, Date: 28.03.2019.
""" Tools to ignore cells with distance greater than some threshold. Can return
list of acitve indices in the covariance matrix, together with corresponding
euclidean distance between the points.

"""
import numpy as np
from sklearn.neighbors import BallTree


class Sparsifier():
    """ Sparsigying functionalities by ignoring cells more distant than a given
    threshold.

    """
    def __init__(self, inverseProblem):
        self.tree = BallTree(inverseProblem.cells_coords)
        self.inverseProblem = inverseProblem


    def get_cells_ind_with_radius(self, cell, radius):
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
        return self.tree.query_radius(cell.reshape(1, -1), radius)[0]

    def compute_sparse_distance_mesh(self, threshold_dist):
        """ Compute the n_model * n_model matrix of squared euclidean distances
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
        # Lists to store the computed squared_dists and indices in the big sparse
        # matrix.
        squared_dists = []
        inds = []

        # Loop over all cells.
        for cell_ind, cell in enumerate(self.inverseProblem.cells_coords):
            print("BallTree processing cell nr: {}".format(cell_ind))

            # Get neighbors.
            neighbor_inds = self.get_cells_ind_with_radius(cell, threshold_dist)

            # We always return at least the cell itself.
            if len(neighbor_inds) == 0:
                neighbor_inds = [cell_ind]

            # Loop over all neighboring cells.
            for neighbor_ind in neighbor_inds:
                squared_dists.append(
                    np.float32(
                        np.linalg.norm(
                            cell - self.inverseProblem.cells_coords[neighbor_ind, :])))
                inds.append((cell_ind, neighbor_ind))
        # Return numpy array, so TF doesnt have to cast later.
        return (np.array(squared_dists), np.array(inds))
