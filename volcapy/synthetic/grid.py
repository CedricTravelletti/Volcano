"""

cells_coords: ndarray, shape n_cells * n_dims

"""
import numpy as np


def buil_cube(nr_x, res_x, nr_y, res_y, nr_z, res_z):
    """ Builds a gridded cube.

    Parameters
    ----------
    nr_x: int
        Number of cells in x-dimension.
    res_x: float
        Size of cell along x_dimension.
    nr_y: int
    res_y: float
    nr_z: int
    res_z: float

    Returns
    -------
    ndarray
        Array of size n_cells * 3.
        Contains the coordinates of the center of each cell.

    """
    # Make sure first cell starts at [0, 0, 0]
    current_x = res_x / 2.0
    current_y = res_y / 2.0
    current_z = res_z / 2.0

    coords = []

    for i in range(nr_x):
        for j in range(nr_y):
            for k in range(nr_z):
                coords.append([current_x, current_y, current_z])
                current_z += res_z
            current_y += res_y
        current_x += res_x
    return np.array(coords)
