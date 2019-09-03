""" Layer between Niklas and us.
Niklas codes include cells far awy from volcano for regularization, we want to
delete those.

We also want to delete the bottom cells (maybe we shouldnt, lets see later).

To effectively delete the cells, we just put the corresponding elements of the
forward to 0, which prevents changing the size of the arrays.

"""
import numpy as np


def get_regularization_cells_inds(inverseProblem):
    """ Get the indices of the troublesome cells in Niklas grid that we want to
    exclude.

    Those are the cells at the edge of the grid and (temporarily) the bottom
    cells.

    Parameters
    ----------
    inverseProblem: InverseProblem

    Returns
    -------
    array[int]
        Indices (in the grid) of the problematic cells.

    """
    # Find the cells at the edges, those are the ones we want to delete.
    max_x = np.max(inverseProblem.cells_coords[:, 0])
    min_x = np.min(inverseProblem.cells_coords[:, 0])
    
    max_y = np.max(inverseProblem.cells_coords[:, 1])
    min_y = np.min(inverseProblem.cells_coords[:, 1])
    
    min_z = np.min(inverseProblem.cells_coords[:, 2])
    
    ind_max_x = np.where(inverseProblem.cells_coords[:, 0] >= max_x)[0]
    ind_min_x = np.where(inverseProblem.cells_coords[:, 0] <= min_x)[0]
    ind_max_y = np.where(inverseProblem.cells_coords[:, 1] >= max_y)[0]
    ind_min_y = np.where(inverseProblem.cells_coords[:, 1] <= min_y)[0]
    ind_min_z = np.where(inverseProblem.cells_coords[:, 2] <= min_z)[0]

    inds = np.concatenate([ind_max_x, ind_min_x, ind_max_y, ind_min_y,
            ind_min_z], axis=0)

    return inds
