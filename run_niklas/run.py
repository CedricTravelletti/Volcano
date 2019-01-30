""" Run current package functionalities on Niklas's data.
"""
from volcapy import loading
from volcapy.niklas.dsm import DSM
from volcapy.niklas.inversion_grid import InversionGrid
import volcapy.niklas.forward as fwd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# LOAD DATA
data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"
dsm = DSM.from_matfile(data_path)

# Coarsening parameters.
# For each inversion cell, it gives the number of DEM cells that are aggregated
# to build it.
coarsen_x = [50] + 194*[5] + [70]
coarsen_y = [50] + 190*[5] + [75]

# Resolution of each inversion cell (in meters).
# Note that inversion cells are coarser than the DEM.
res_x = [5000] + 194*[50] + [7000]
res_y = [5000] + 190*[50] + [7500]

# Resolution of the DEM cells.
dem_res_x = 50*[100] + 5*194*[10] + 70*[100]
dem_res_y = 50*[100] + 5*190*[10] + 75*[100]

z_base = -5000
z_low = -475
z_high = 925
z_step = 50

# Create a floor every *step*. We have to add 1 so that the z_high is included
# in the interval.
# TODO: May be bug prone: we have to cast to list, since arange produces an
# array.
z_levels = list(np.arange(z_low, z_high + 1, z_step))

# Add the lowest level.
z_levels = [z_base] + z_levels

# Create the inversion grid.
inversion_grid = InversionGrid(
        coarsen_x, coarsen_y, res_x, res_y, z_levels, dsm)

# Load the data points coordinates.
data = loading.load_niklas(data_path)
data_coords = data['data_coords']

# Build the Forward.
F = fwd.forward(inversion_grid, data_coords, z_base)


"""
# Plotting.

# Subsample some points.
nx = 1000
ny = 1000

x = []
y = []
z = []

for i in range(nx):
    for j in range(ny):
        coords = dsm[i, j]
        x.append(coords[0])
        y.append(coords[1])
        z.append(coords[2])

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
        cmap='viridis', edgecolor='none')
"""
