# File: sparsifier.py, Author: Cedric Travelletti, Date: 28.03.2019.
""" Tools to ignore cells with distance greater than some threshold. Can return
list of acitve indices in the covariance matrix, together with corresponding
euclidean distance between the points.

"""
import numpy as np
from sklearn.neighbors import BallTree
from multiprocessing import Pool
from itertools import repeat, chain


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
        """ Compute the n_model * n_model matrix of euclidean distances
        between all couples, ignoring couples with a distance greather than
        *threshold_dist*.

        Parameters
        ----------
        threshold_dist: float
            Only consider couples with distance less than this.

        Returns
        -------
        dists: List[float]
            List of euclidean distances between couples of cells.
        inds: List[(int, int)]
            Gives the indices inside the full matrix, of the values in the
            preceding list.

        """
        pool = Pool()
        # Loop (in parallel) over all cells (here corresponds to looping ove
        # rows of the covariace/mesh matrix.
        # For each cell/row, only consider elements of the covariance/mesh
        # matrix that are within a certain threshold_dist.

        # The code below is a little bit involved, since it uses
        # multiprocessing. Python multiprocessing only allows to loop in
        # parallele over a single list and return a single list.
        # We thus have to unpack the output using the zip(* ) idiom and to pack
        # the inputs using zip and the itertools.repeat() to give the same
        # threshold_dist to each process.

        # Finally, since a single function call (on on cell) already returns a
        # list, we will get lists of lists. We unpack them using
        # itertools.chain.
        dists, inds = zip(
            *pool.starmap(self.per_cell_dist_mesh,
                    zip(self.inverseProblem.cells_coords,
                            list(range(self.inverseProblem.n_model)),
                            repeat(threshold_dist))))
        # Free up resources.
        pool.close()
        pool.join()

        # Un-nest the lists using itertools.chain.
        dists = list(chain.from_iterable(dists))
        inds = list(chain.from_iterable(inds))

        # Return numpy array, so TF doesnt have to cast later.
        return (np.array(dists), np.array(inds))

    def per_cell_dist_mesh(self, cell, cell_ind, threshold_dist):
        """ Helper function for the above, so we can parallelize.

        Parameters
        ----------
        cell: Cell
            Cells to consider in the covariance matrix.
        cell_ind: int
            Index of the cell in the covariance matrix (its row number).

        Returns
        -------

        """
        # Lists to store the computed squared_dists and indices in the big sparse
        # matrix.
        dists = []
        inds = []

        print("BallTree processing cell nr: {}".format(cell_ind))

        # Get neighbors.
        neighbor_inds = self.get_cells_ind_with_radius(cell, threshold_dist)

        # We always return at least the cell itself.
        if len(neighbor_inds) == 0:
            neighbor_inds = [cell_ind]

        # Loop over all neighboring cells.
        for neighbor_ind in neighbor_inds:
            dists.append(
                np.float32(
                    np.linalg.norm(
                        cell - self.inverseProblem.cells_coords[neighbor_ind, :])))
            inds.append((cell_ind, neighbor_ind))
        return (dists, inds)
