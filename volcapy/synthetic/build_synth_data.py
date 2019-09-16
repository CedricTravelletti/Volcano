import volcapy.synthetic.grid as gd
import numpy as np
import os


# Generate gridded cube.
nx = 50
ny = 50
nz = 50
res_x = 1
res_y = 1
res_z = 1
coords = gd.build_cube(nx, res_x, ny, res_y, nz, res_z)

# Put evenly spaced measurement sites on the surface of the cube.
max_x = np.max(coords[:, 0])

data_coords = gd.generate_regular_surface_datapoints(
        0.0, max_x, 5, 0.0, max_x, 5, 0.0, max_x, 5, offset=0.1)

# Compute the forward operator.
F = gd.compute_forward(coords, res_x, res_y, res_z, data_coords)

# Put matter inside the cube.
density = np.zeros((coords.shape[0],))
density[:] = 1500.0

density[(
        (coords[:, 0] > 10) & (coords[:, 0] < 20)
        & (coords[:, 1] > 20) & (coords[:, 1] < 22)
        & (coords[:, 2] > 10) & (coords[:, 2] < 40))] = 2400.0

# density[0] = 2800

"""
density[(
        (coords[:, 0] > 10) & (coords[:, 0] < 20)
        & (coords[:, 1] > 20) & (coords[:, 1] < 22)
        & (coords[:, 2] > 10) & (coords[:, 2] < 40))] = 2000.0

density[0] = 2800
"""


# Generate artificial measurements.
data_values = F @ density

# Save
out_folder = "./out/"
np.save(os.path.join(out_folder, "F_synth.npy"), F)
np.save(os.path.join(out_folder,"coords_synth.npy"), coords)
np.save(os.path.join(out_folder,"data_coords_synth.npy"), data_coords)
np.save(os.path.join(out_folder,"data_values_synth.npy"), data_values)
np.save(os.path.join(out_folder,"density_synth.npy"), density)

# -------------------------------------------------------------------
# Save to VTK for alter visualiation with Paraview.
# -------------------------------------------------------------------
from volcapy.synthetic.vtkutils import save_vtk

save_vtk(density, (nx, ny, nz), res_x, res_y, res_z, os.path.join(out_folder, "density_synth.mhd"))
