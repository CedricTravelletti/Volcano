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
posterior_mean_path_exp = "/home/cedric/PHD/run_results/forwards/m_post_902_exponential.npy"
m_post_m_exp = np.load(posterior_mean_path_exp)

posterior_mean_path_sqexp = "/home/cedric/PHD/run_results/forwards/m_post_342_squared_exponential.npy"
m_post_m_sqexp = np.load(posterior_mean_path_sqexp)

posterior_mean_path_matern32 = "/home/cedric/PHD/run_results/forwards/m_post_562_matern32.npy"
m_post_m_matern32 = np.load(posterior_mean_path_matern32)

posterior_mean_path_matern52 = "/home/cedric/PHD/run_results/forwards/m_post_462_matern52.npy"
m_post_m_matern52 = np.load(posterior_mean_path_matern52)

# SPECIAL: Remove regularisation cells.
regularisation_inds = get_regularization_cells_inds(inverseProblem)
m_post_m_exp = np.delete(m_post_m_exp, regularisation_inds)
m_post_m_sqexp = np.delete(m_post_m_sqexp, regularisation_inds)
m_post_m_matern32 = np.delete(m_post_m_matern32, regularisation_inds)
m_post_m_matern52 = np.delete(m_post_m_matern52, regularisation_inds)

# Match to regular grid.
reg_inds, reg_coords, coords, grid_metadata = match_grids(inverseProblem)
nx = grid_metadata['nx']
ny = grid_metadata['ny']
nz = grid_metadata['nz']


# Important: Cubic regular array containing cell inds.
cubic_reg_coords = reg_coords.reshape(nx, ny, nz, 3)

res_x = 50
res_y = 50
res_z = 50

# Put density values in regular grid.
m_post_reg_exp = np.zeros(reg_coords.shape[0])
m_post_reg_sqexp = np.zeros(reg_coords.shape[0])
m_post_reg_matern32 = np.zeros(reg_coords.shape[0])
m_post_reg_matern52 = np.zeros(reg_coords.shape[0])
# m_post_reg[reg_inds] = m_post_m.reshape(-1)

m_post_reg_exp[reg_inds] = m_post_m_exp
m_post_reg_exp = m_post_reg_exp.reshape(nx, ny, nz)

m_post_reg_sqexp[reg_inds] = m_post_m_sqexp
m_post_reg_sqexp = m_post_reg_sqexp.reshape(nx, ny, nz)

m_post_reg_matern32[reg_inds] = m_post_m_matern32
m_post_reg_matern32 = m_post_reg_matern32.reshape(nx, ny, nz)

m_post_reg_matern52[reg_inds] = m_post_m_matern52
m_post_reg_matern52 = m_post_reg_matern52.reshape(nx, ny, nz)

# Height of the lowest level.
z_base = -450
z_index_offset = 9

import matplotlib.pyplot as plt

# Remove non-volcano cells from the regular grid.
m_post_reg_exp[m_post_reg_exp<=10.0] = np.nan
m_post_reg_sqexp[m_post_reg_sqexp<=10.0] = np.nan
m_post_reg_matern32[m_post_reg_matern32<=10.0] = np.nan
m_post_reg_matern52[m_post_reg_matern52<=10.0] = np.nan


"""
# -----------------------------
# Slice z=0.
# -----------------------------

ax = plt.subplot(221)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(222)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(223)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(224)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Gaussian")
ax.imshow(m_post_reg_sqexp[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

plt.show()

# -----------------------------
# Slice z=500.
# -----------------------------

ax = plt.subplot(221)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(222)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(223)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(224)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Gaussian")
ax.imshow(m_post_reg_sqexp[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

plt.show()

# -----------------------------
# Slice z=-150.
# -----------------------------

ax = plt.subplot(221)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(222)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(223)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = plt.subplot(224)
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_title("Gaussian")
ax.imshow(m_post_reg_sqexp[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

plt.show()
"""

from mpl_toolkits.axes_grid1 import AxesGrid


# -----------------------------
# Slice z=-150.
# -----------------------------

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
ax = grid[0]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[1]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[2]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[3]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Gaussian")
im = ax.imshow(m_post_reg_sqexp[:, :, 6].T, cmap="jet", vmin=1750, vmax=2650)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

plt.show()

# -----------------------------
# Slice z=0.
# -----------------------------

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
ax = grid[0]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[1]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[2]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[3]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Gaussian")
im = ax.imshow(m_post_reg_sqexp[:, :, 9].T, cmap="jet", vmin=1750, vmax=2650)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

plt.show()

# -----------------------------
# Slice z=500.
# -----------------------------

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
ax = grid[0]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[1]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[2]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[3]
ax.set_ylim((19, 124))
ax.set_xlim((19, 124))
ax.set_axis_off()
ax.set_title("Gaussian")
im = ax.imshow(m_post_reg_sqexp[:, :, 19].T, cmap="jet", vmin=1750, vmax=2650)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

plt.show()

# -----------------------------
# -----------------------------
# -----------------------------
# Y SLICE.
# -----------------------------
# -----------------------------
# -----------------------------
from volcapy.compatibility_layer import get_regularization_cells_inds
reg_cells_inds = get_regularization_cells_inds(inverseProblem)

# Longitude of the slice.
# NB, see below for how to find closest cell latitude to actual slice latitude.
y_slice = inverseProblem.cells_coords[np.where(
        np.abs(inverseProblem.cells_coords[:, 1] -4293.25*1000)<30)[0], 1][0]

# Origin of the grid (have to remove regularisation cells).
cells_wh_reg = np.delete(inverseProblem.cells_coords, reg_cells_inds, axis=0)
y_orig = np.min(cells_wh_reg[:, 1])
y_ind = int((y_slice - y_orig)/res_y)

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.4,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
ax = grid[0]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[:, y_ind, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[1]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[:, y_ind, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[2]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[:, y_ind, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[3]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Gaussian")
im = ax.imshow(m_post_reg_sqexp[:, y_ind, :].T, cmap="jet", vmin=1750, vmax=2650)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

plt.show()


# -----------------------------
# -----------------------------
# -----------------------------
# X SLICE.
# -----------------------------
# -----------------------------
# -----------------------------

# Longitude of the slice.
# NB, see below for how to find closest cell latitude to actual slice latitude.
x_slice = inverseProblem.cells_coords[np.where(
        np.abs(inverseProblem.cells_coords[:, 0] -518.75*1000)<30)[0], 0][0]

x_orig = np.min(cells_wh_reg[:, 0])
x_ind = int((x_slice - x_orig)/res_x)

fig = plt.figure()
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.4,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
ax = grid[0]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Exponential")
ax.imshow(m_post_reg_exp[x_ind, :, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[1]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Matérn 3/2")
ax.imshow(m_post_reg_matern32[x_ind, :, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[2]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Matérn 5/2")
ax.imshow(m_post_reg_matern52[x_ind, :, :].T, cmap="jet", vmin=1750, vmax=2650)

ax = grid[3]
ax.set_ylim((0, 30))
ax.set_xlim((0, 130))
ax.set_axis_off()
ax.set_title("Gaussian")
im = ax.imshow(m_post_reg_sqexp[x_ind, :, :].T, cmap="jet", vmin=1750, vmax=2650)

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

plt.show()
