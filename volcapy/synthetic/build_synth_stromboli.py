""" Build synthetic forward and data for the real Stromboli data of Niklas.

The goal is to get rid of the outliers in Niklas's forward.

""" 
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import get_regularization_cells_inds
import volcapy.synthetic.grid as gd
import volcapy.covariance.matern32 as cl

import numpy as np
import os


def main():
    # ----------------------------------------------------------------------------#
    #      LOAD NIKLAS DATA
    # ----------------------------------------------------------------------------#
    # Initialize an inverse problem from Niklas's data.
    # This gives us the forward and the coordinates of the inversion cells.
    niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
    # niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
    # niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
    inverseProblem = InverseProblem.from_matfile(niklas_data_path)
    
    # -- Delete Regularization Cells --
    # Delete the cells.
    # reg_cells_inds = get_regularization_cells_inds(inverseProblem)
    # inverseProblem.forward[:, reg_cells_inds] = 0.0
    volcano_coords = inverseProblem.cells_coords
    data_coords = inverseProblem.data_points

    # Compute the forward operator.
    res_x, res_y, res_z = 50, 50, 50
    F_synth = gd.compute_forward(volcano_coords, res_x, res_y, res_z, data_coords)
    # print(F_synth)

    # Generate artificial measurements.
    # data_values = F_synth @ irreg_density

if __name__ == "__main__":
    main()
