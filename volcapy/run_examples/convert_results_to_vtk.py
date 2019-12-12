""" Convert the results to the VTK file format for plotting using ParaView.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
from volcapy.compatibility_layer import match_grids, get_regularization_cells_inds
from volcapy.synthetic.vtkutils import ndarray_to_vtk

import numpy as np
import os

niklas_data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
# niklas_data_path = "/home/ubuntu/Volcano/data/Cedric.mat"
# niklas_data_path = "/idiap/temp/ctravelletti/tflow/Volcano/data/Cedric.mat"
inverseProblem = InverseProblem.from_matfile(niklas_data_path)


# Load inversion results.
posterior_mean_path = "/home/cedric/PHD/run_results/forwards/m_post_342_squared_exponential.npy"
m_post_m = np.load(posterior_mean_path)

# SPECIAL: Remove regularisation cells.
reg_cells_inds, bottom_inds = get_regularization_cells_inds(inverseProblem)
inds_to_delete = list(set(
        np.concatenate([reg_cells_inds, bottom_inds], axis=0)))
m_post_m = np.delete(m_post_m, inds_to_delete)

# Match to regular grid.
reg_inds, reg_coords, coords, grid_metadata = match_grids(inverseProblem)
nx = grid_metadata['nx']
ny = grid_metadata['ny']
nz = grid_metadata['nz']

res_x = 50
res_y = 50
res_z = 50

# Put density values in regular grid.
m_post_reg = np.zeros(reg_coords.shape[0])
# m_post_reg[reg_inds] = m_post_m.reshape(-1)
m_post_reg[reg_inds] = m_post_m
m_post_reg = m_post_reg.reshape(nx, ny, nz)

ndarray_to_vtk(m_post_reg, res_x, res_y, res_z,
            "reconstructed_density_cholesky.mhd")
