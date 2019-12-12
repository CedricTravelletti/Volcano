""" Layer between Niklas and us.
Niklas codes include cells far awy from volcano for regularization, we want to
delete those.

We also want to delete the bottom cells (maybe we shouldnt, lets see later).

To effectively delete the cells, we just put the corresponding elements of the
forward to 0, which prevents changing the size of the arrays.

"""
import volcapy.synthetic.grid as gd
import numpy as np


def get_regularization_cells_inds(inverseProblem):
    """ Get the indices of the troublesome cells in Niklas grid that we want to
    exclude.

    Those are the (bigger) cells at the edge of the grid.
    We also return the indices of the bottom cells.

    Parameters
    ----------
    inverseProblem: InverseProblem

    Returns
    -------
    (reg_cells_inds, bottom_cells_ind)
    array[int]
        Indices (in the grid) of the problematic/regularization cells.
    array[int]
        Indices (in the grid) of the bottom cells.

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

    # Find bottom cells.
    ind_min_z = np.where(inverseProblem.cells_coords[:, 2] <= min_z)[0]

    reg_cells_inds = np.concatenate([ind_max_x, ind_min_x, ind_max_y, ind_min_y], axis=0)
    bottom_cells_inds = ind_min_z

    return (reg_cells_inds, bottom_cells_inds)

def match_grids(inverseProblem):
    # Get normal cells only (so we have a regular grid).
    reg_cells_inds, bottom_cells_inds = get_regularization_cells_inds(inverseProblem)
    inds_to_delete = list(set(
            np.concatenate([reg_cells_inds, bottom_cells_inds], axis=0)))

    coords = np.delete(inverseProblem.cells_coords,
            inds_to_delete, axis=0)

    min_x = np.min(coords[:, 0])
    max_x = np.max(coords[:, 0])
    min_y = np.min(coords[:, 1])
    max_y = np.max(coords[:, 1])
    min_z = np.min(coords[:, 2])
    max_z = np.max(coords[:, 2])

    res_x = 50
    res_y = 50
    res_z = 50

    # Check resolution compatible with grid extent.
    if (((max_x - min_x) % res_x == 0)
            and ((max_y - min_y) % res_y == 0)
            and ((max_z - min_z) % res_z == 0)):
        nx = int((max_x - min_x) / res_x) + 1
        ny = int((max_y - min_y) / res_y) + 1
        nz = int((max_z - min_z) / res_z) + 1
    else:
        raise ValueError(
            "Grid extent not divisible by resolution -> unregular grid.")

    # Build coresponding regular grid.
    reg_coords = gd.build_cube(nx, res_x, ny, res_y, nz, res_z)

    # Shift cells so the begin at zero.
    # Note, since we are using centroids, we have to move past zero.
    coords[:, 0] = coords[:, 0] - min_x + res_x / 2.0
    coords[:, 1] = coords[:, 1] - min_y + res_y / 2.0
    coords[:, 2] = coords[:, 2] - min_z + res_z / 2.0

    reg_inds = np.zeros(coords.shape[0], dtype=np.int)
    for i, cell in enumerate(coords):
        reg_inds[i] = index_in_reg_grid(cell, nx, ny, nz, res_x, res_y, res_z)

    # Return some metadata.
    grid_metadata = {'nx': nx, 'ny': ny, 'nz': nz}
    return reg_inds, reg_coords, coords, grid_metadata

def index_in_reg_grid(cell, nx, ny, nz, res_x, res_y, res_z):
    """ Given coordinates, find index in regular array.

    """
    # Correct for centroid.
    x = cell[0] - res_x / 2.0
    y = cell[1] - res_y / 2.0
    z = cell[2] - res_z / 2.0
    ind = (x / res_x) * (ny * nz) + (y / res_y) * nz + (z / res_z)

    if not ind.is_integer():
        raise ValueError("Non-integer index. Resolutions must be wrong.")
    return int(ind)
