""" File: grid.py, Author: Cedic Travelletti, Date: 02.12.2018.

Build covariance matrix implicitly.
Idea is to have K[i, j] = f(i, j), so matrix is built on the fly.
The function f is expansive to compute so we use a hash table.
Tricky part is the mapping (i, j) -> f(i, j).

"""
import numpy as np


def buil_hash_grid(grid, sigma_2, lambda_2):
    """ Built a list of the pre-computed covariance kernel
    on the grid.

    """
    covariance_hash = np.zeros(len(grid))

    # Subtract coords of the origin.
    grid = np.array(grid)
    orig_x = np.min(grid[:, 0])
    orig_y = np.min(grid[:, 1])
    orig_z = np.min(grid[:, 2])

    grid = grid - [orig_x, orig_y, orig_z]

    for i in range(len(grid)):
        dist = grid[i][0]**2 + grid[i][1]**2 + grid[i][2]**2
        covariance_hash[i] = sigma_2 * np.exp(-dist / lambda_2)

    return covariance_hash


def regularize_index(i, grid, orig_x, orig_y, orig_z,
        dx, dy, dz):
    """ Find index in a regular grid of spacing dx, dy, dz.

    Parameters
    ----------
    orig_x: float
        Origin of coordinates.
    orig_y: float
    orig_z: float
    dx: float
        Spacing between cells (regular).
    dy: float
    dz: float

    """

"""
# Transform 2D index into 1D.
np.ravel_multi_index((i, j), (dim_i, dim_j))

# Transform 1D-index into 2D.
np.unravel_index(ind_1d, (dim_i, dim_j))
"""


def build_regular_grid(xmin, xmax, dx,
        ymin, ymax, dy,
        zmin, zmax, dz):
    """ Build a regular rectangular grid.

    Parameters
    ----------
    xmin: float
        Smallest x coord in the grid.
    xmax: float
        Biggest x coord in the grid (included).
    dx: float
        Spacing between two cells.
    ymin: float
    ymax: float
    dy: float
    zmin: float
    zmax: float
    dz: float

    Returns
    -------
    [float, float, float]
        Coordinates of the grid. Ordering: [x, y, z].

    """
    nx = round((xmax - xmin) / dx) + 1
    ny = round((ymax - ymin) / dy) + 1
    nz = round((zmax - zmin) / dz) + 1

    grid = []

    for k in np.linspace(zmin, zmax, nz, endpoint=True):
        for j in np.linspace(ymin, ymax, ny, endpoint=True):
            for i in np.linspace(xmin, xmax, nx, endpoint=True):
                grid.append([i, j, k])

    return grid

def regularize_grid(grid, dx, dy, dz):
    """ Given an sparse regular grid, build the smallest full regular grid
    containing it.

    Parameters
    ----------
    grid: List[[float, float, float]]
        List of the coordinates of every point in the grid.
    dx: float
        (regular) spacing between the cells in the x direction.
    dy:float
    dz: float

    Returns
    -------
    List[[float, float, float]]
        List of coordinates of the points of the full grid.
        The list is ordered, looping furst through x ,then through y, then
        through z.

    """
    grid = np.array(grid)
    xmin = np.amin(grid[:, 0])
    xmax = np.amax(grid[:, 0])
    ymin = np.min(grid[:, 1])
    ymax = np.max(grid[:, 1])
    zmin = np.min(grid[:, 2])
    zmax = np.max(grid[:, 2])

    return build_regular_grid(xmin, xmax, dx,
                              ymin, ymax, dy,
                              zmin, zmax, dz)

