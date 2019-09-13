"""

cells_coords: ndarray, shape n_cells * n_dims

"""
import numpy as np
from volcapy.niklas.banerjee import banerjee


def build_cube(nr_x, res_x, nr_y, res_y, nr_z, res_z):
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
    orig_x = res_x / 2.0
    orig_y = res_y / 2.0
    orig_z = res_z / 2.0

    coords = []

    current_x = orig_x
    for i in range(nr_x):
        current_y = orig_y
        for j in range(nr_y):
            current_z = orig_z
            for k in range(nr_z):
                coords.append([current_x, current_y, current_z])
                current_z += res_z
            current_y += res_y
        current_x += res_x
    return np.array(coords)

def compute_forward(coords, res_x, res_y, res_z, data_coords):
    """ Compute forward operator.

    Parameters
    ----------
    coords: ndarray
        Cells centroid coordinates, size n_cell * n_dims.
    res_x: float
        Length of a cell in x-direction (meters).
    res_y_float
    res_z: float
    data_coords: ndarray
        List of data measurements coordinates, size n_data * n_dims.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    n_cells = coords.shape[0]
    n_data = data_coords.shape[0]
    F = np.zeros((n_data, n_cells))

    for i, cell in enumerate(coords):
        for j, data in enumerate(data_coords):
            # Compute cell endpoints.
            xh = cell[0] + res_x / 2.0
            xl = cell[0] - res_x / 2.0
            yh = cell[1] + res_y / 2.0
            yl = cell[1] - res_y / 2.0
            zh = cell[2] + res_z / 2.0
            zl = cell[2] - res_z / 2.0
            
            F[j, i] = banerjee(
                    xh, xl, yh, yl, zh, zl,
                    data[0],  data[1],  data[2])
    return F

def generate_regular_surface_datapoints(
        xl, xh, nx, yl, yh, ny, zl, zh, nz,
        offset):
    """ Put regularly spaced measurement points on the surface of a
    cube.
    Note that there will always be measurement sites at the endpoints of the
    cube.
    We need an offset because measerements cannot be directly on the endpoints
    of a cell because of division by zero in the Bannerjee formula.

    Parameters
    ----------
    xl: float
        Lower x-coordinate of the cube.
    xh: float
        Higher x-coordinate of the cube.
    nx: int
        Number of measurments in x-dimension.
    yl: float
    yh: float
    ny: int
    zl: float
    zh: float
    nz: int
    offset: float
        Displace the measurements sites by an offset outside of the cube to
        avoid division by zero.

    Returns
    -------
    ndarray
        Coordinates of the measurement sites, size n_data * n_dims.

    """
    data_coords = []

    # Bottom x surface.
    for y in np.linspace(yl - offset, yh + offset, ny):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([xl - offset, y, z])

    # Top x surface.
    for y in np.linspace(yl - offset, yh + offset, ny):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([xh + offset, y, z])

    # Bottom y surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([x, yl - offset, z])

    # Top y surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for z in np.linspace(zl - offset, zh + offset, nz):
            data_coords.append([x, yh + offset, z])

    # Bottom z surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for y in np.linspace(yl - offset, yh + offset, ny):
            data_coords.append([x, y, zl - offset])

    # Top z surface.
    for x in np.linspace(xl - offset, xh + offset, nx):
        for y in np.linspace(yl - offset, yh + offset, ny):
            data_coords.append([x, y, zh + offset])
    return np.array(data_coords)
