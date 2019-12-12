""" Dumps volcano cells coords and measurements coords to txt.

Goal is then to use it with the VTK visualization script.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.compatibility_layer import get_regularization_cells_inds

import numpy as np
import os


def main():
    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    # niklas_data_path = "/home/ubuntu/Dev/Data/Cedric.mat"
    # niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)

    # -- Delete Regularization Cells --
    # Delete the cells.
    reg_cells_inds = get_regularization_cells_inds(inverseProblem)
    inverseProblem.forward[:, reg_cells_inds] = 0.0

    F = inverseProblem.forward
    d_obs = inverseProblem.data_values
    data_coords = inverseProblem.data_points
    cells_coords = np.delete(inverseProblem.cells_coords, reg_cells_inds,
            axis=0)
    np.savetxt("data_coords.txt", data_coords, fmt="%4f", delimiter=" ", header="data\nx y z")
    np.savetxt("volcano_coords.txt", cells_coords, fmt="%4f", delimiter=" ", header="data\nx y z")

if __name__ == "__main__":
    main()
    
