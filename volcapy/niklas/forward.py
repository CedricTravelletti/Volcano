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

def compute_cell_response_at_point(cell, point,
        is_topcell=False, z_base=0):
    """ Compute the repsonse of an individual inversion cell on a measurement
    point.

    Parameters
    ----------
    cell: Cell
        Inversion cell whose response we want to compute.
    point: (float, float, float)
        Coordinates (x, y, z) of the point at which we measure the response.
    is_topcell: bool
        Should set to true when computing the repsonse of a top cell.
        Such cells are treated differently, in that we take the prism to be
        from the elevation, down to the base.
        If true, then the z_base argument should be provided.
    z_base: float
        Altitude (in meters) of the lowest level we consider.

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

    # Special treatment for top cells: we go from their altitude down to the
    # base.
    if is_topcell:
        zl = z_base
        zh = cell.z

    return banerjee(xh, xl, yh, yl, zh, zl,
            point[0], point[1], point[2])

# TODO: Currently modifies the operator in place.
# Might be good to make it side-effect free.
def correct_forward(F, inversion_grid, data_points, z_base):
    """ Correct the forward at the topmost cells
    Parameters
    ----------
    F: array-like
        Forward we want to correct.
    inversion_grid: InversionGrid
    data_points: List[(float, float, float)]

    """

    # Only loop over cells that are on the surface (i.e. 'topmost' ones).
    for i in inversion_grid.topmost_indices:
        print(i)

        # Get the fine cells that describe the topography at the current
        # (coarse) cell. Compute each response and add up.
        fine_cells = inversion_grid.fine_cells_from_topmost_ind(i)

        for j, point in enumerate(data_points):
            # Add the responses from each fine cell.
            temp_F = 0
            for fine_cell in fine_cells:
                temp_f += compute_cell_response_at_point(fine_cell, point,
                    True, z_base)

        F[i, j] = temp_F

        return F
