# File: forward.py, Author: Cedric Travelletti, Date: 17.01.2019.
""" Compute forward operator for a whole inversion grid.

"""
import numpy as np
from volcapy.niklas.inversion_grid import InversionGrid
from volcapy.niklas.banerjee import banerjee


def forward(inversion_grid, data_points, z_base):
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
    z_base: float
        Altitude (in meters) of the lowest level we consider. I.e., we will
        build inversion cells down to that level.

    """
    n_cells = len(inversion_grid)
    n_data = len(data_points)

    F = np.zeros((n_cells, n_data))

    for i, cell in enumerate(inversion_grid):
        print(i)
        # If it is a top cell, then it contains refinement attributes, so we
        # compute differently.
        if cell.is_topcell:
            print("Top")
            for j, point in enumerate(data_points):
                F[i, j] = compute_top_cell_response_at_point(cell, point,
                        z_base=z_base)

        else:
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


def compute_top_cell_response_at_point(cell, point, z_base=0):
    """ Same as the above, but for a cell which contains subdivisions (i.e. a
    topmost cell).
    Note that for such cells, we extend the parallelograms down to zbase.

    Parameters
    ----------
    cell: Cell
        Inversion cell whose response we want to compute.
    point: (float, float, float)
        Coordinates (x, y, z) of the point at which we measure the response.
    z_base: float
        Altitude (in meters) of the lowest level we consider.

    """
    # Loop over the subdivisions.
    F = 0.0
    for subcell in cell.fine_cells:
        # Define the corners of the parallelepiped.
        # We consider the x/y of the cell to be in the middle, so we go one
        # half resolution to the left/right.
        xh = subcell.x + subcell.res_y/2
        xl = subcell.x - subcell.res_y/2

        yh = subcell.y + subcell.res_y/2
        yl = subcell.y - subcell.res_y/2

        zl = z_base
        zh = subcell.z

        # Add the contributions of each subcells.
        F += banerjee(xh, xl, yh, yl, zh, zl,
                point[0], point[1], point[2])
    return F
