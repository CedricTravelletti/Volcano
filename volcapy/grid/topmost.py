""" Utility for findig the surface cells in a grid.

Main use is when lowering the resolution: we want to treat the top cells
independently, since thy contribute more.

"""
from volcapy.inverse.flow import InverseProblem
import numpy as np

# Initialize an inverse problem from Niklas's data.
# This gives us the forward and the coordinates of the inversion cells.
niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)


def gridify(cells_coords, base_resolution):
    """ Try to find a regular grid from which the cells might come.

    Here, the cells must come from a regular grid with a given base resolution,
    to which we have added a buffer of border cells. (Maximum one row/column of
    border cell per side.)

    """
    min_x = np.min(cells_coords[:, 0])
    max_x = np.max(cells_coords[:, 0])
    min_y = np.min(cells_coords[:, 1])
    max_y = np.max(cells_coords[:, 1])
    min_z = np.min(cells_coords[:, 2])
    max_z = np.max(cells_coords[:, 2])

    # Find cells that are at the border. Note that we dont consider cells that
    # are at the positive z-border (top).
    border_inds = np.where((cells_coords[:, 0] <= min_x) | (cells_coords[:, 0] >= max_x) | 
            (cells_coords[:, 1] <= min_y) | (cells_coords[:, 1] >= max_y) |
            (cells_coords[:, 2] <= min_z))[0]
