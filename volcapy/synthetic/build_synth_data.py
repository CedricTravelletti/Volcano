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

# Put matter in a cone.
cone_inds, surface_inds = gd.build_random_cone(coords, nx, ny, nz)

# Discard cells that are not in the cone when building the forward.
volcano_coords = coords[cone_inds]

# Put matter inside the volcano.
density = np.zeros((coords.shape[0],), dtype=np.float32)

# Note that we keep the regular grid arrays for visualization purposes, but use
# irregular arrays (with only the volcano cells) during inversion to make it
# lighter.
irreg_density = density[cone_inds]
irreg_density[:] = 1500

# Add an overdensity.
irreg_density[(
        (volcano_coords[:, 0] > 10) & (volcano_coords[:, 0] < 20)
        & (volcano_coords[:, 1] > 20) & (volcano_coords[:, 1] < 22)
        & (volcano_coords[:, 2] > 10) & (volcano_coords[:, 2] < 40))] = 2000.0

density[cone_inds] = irreg_density
"""
data_coords = gd.generate_regular_surface_datapoints(
        0.0, max_x, 5, 0.0, max_x, 5, 0.0, max_x, 5, offset=0.1)
"""
# -------
# WARNING
# -------
# We put measurements close to the surface by randomly selecting surface cells
# and adding a small vertical shift.
n_data = 20
data_inds = np.random.choice(surface_inds, n_data, replace=False)
data_coords = coords[data_inds]

offset = 0.05 * res_z
data_coords[:, 2] = data_coords[:, 2] + offset

# Compute the forward operator.
F = gd.compute_forward(volcano_coords, res_x, res_y, res_z, data_coords)

# Generate artificial measurements.
data_values = F @ irreg_density

# Save
out_folder = "./out/"
np.save(os.path.join(out_folder, "F_synth.npy"), F)
np.save(os.path.join(out_folder,"coords_synth.npy"), coords)
np.save(os.path.join(out_folder,"volcano_inds_synth.npy"), cone_inds)
np.save(os.path.join(out_folder,"data_coords_synth.npy"), data_coords)
np.save(os.path.join(out_folder,"data_values_synth.npy"), data_values)
np.save(os.path.join(out_folder,"density_synth.npy"), density)

# -------------------------------------------------------------------
# Save to VTK for alter visualiation with Paraview.
# -------------------------------------------------------------------
from volcapy.synthetic.vtkutils import save_vtk

save_vtk(density, (nx, ny, nz), res_x, res_y, res_z,
        os.path.join(out_folder, "density_synth.mhd"))

# Also save a grid with location of the measurements.
data_sites_reg = np.zeros(coords.shape[0])
data_sites_reg[data_inds] = 1
save_vtk(data_sites_reg, (nx, ny, nz), res_x, res_y, res_z,
        os.path.join(out_folder, "data_sites_synth.mhd"))
