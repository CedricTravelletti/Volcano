# File: forward.py, Author: Cedric Travelletti, Date: 17.01.2019.
""" Compute forward operator for a whole inversion grid.

"""
import numpy as np
from volcapy.niklas.inversion_grid import InversionGrid
from volcapy.niklas.banerjee import banerjee


def forward(inversion_grid, data_points):
    """ Compute forward operator associated to a given geometry/discretization
    defined by an inversion grid.
    The forward give the response at locations defined by the datapoints
    vector.

    Parameters
    ----------
    inversion_grid: InversionGrid
    data_points: List[(float, float, float)]
        List containing the coordinates, in order (x, y, z) of the data points
        at which we measure the response / gravitational field.

    """
    n_cells = len(inversion_grid)
    n_data = len(data_points)

    F = np.zeros((n_cells, n_data))

    for i, cell in enumerate(inversion_grid):
        print(i)
        for j, point in enumerate(data_points):

            F[i, j] = compute_cell_response_at_point(cell, point)

    return F

def compute_cell_response_at_point(cell, point):
    """ Compute the repsonse of an individual inversion cell on a measurement
    point.

    Parameters
    ----------
    cell: Cell
        Inversion cell whose response we want to compute.
    point: (float, float, float)
        Coordinates (x, y, z) of the point at which we measure the response.

    """

    # Define the corners of the parallelepiped.
    # We consider the x/y of the cell to be in the middle, so we go one
    # half resolution to the left/right.
    xh = cell.x + cell.res_y/2
    xl = cell.x - cell.res_y/2

    yh = cell.y + cell.res_y/2
    yl = cell.y - cell.res_y/2

    # TODO: Warning, z stuff done here, see issues.
    zl = cell.z
    zh = zl + cell.res_z

    return banerjee(xh, xl, yh, yl, zh, zl,
            point[0], point[1], point[2])

def refine_forward(F, inversion_grid, data_points):
    """ Refine an already computed forward. That is, modify the response of the
    topmost cells to take the topography into account.

    Parameters
    ----------
    F: array-like
        Forward we want to correct.
    inversion_grid: InversionGrid
    data_points: List[(float, float, float)]

    """
    n_cells = len(inversion_grid)
    n_data = len(data_points)

    # Only loop over cells that are on the surface (i.e. 'topmost' ones).
    for i in inversion_grid.topmost_indices:
        print(i)
        cell = inversion_grid.cells[i]

        # Get the fine cells that describe the topography at the current
        # (coarse) cell. Compute each response and add up.


        for j, point in enumerate(data_points):
            F[i, j] = compute_cell_response_at_point(cell, point)

    return F
