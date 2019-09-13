import volcapy.synthetic.grid as gd
import numpy as np


# Generate gridded cube.
coords = gd.build_cube(50, 1, 50,1, 50, 1)

# Put evenly spaced measurement sites on the surface of the cube.
max_x = np.max(coords[:, 0])

data_coords = gd.generate_regular_surface_datapoints(
        0.0, max_x, 10, 0.0, max_x, 10, 0.0, max_x, 10, 0.1)

# Compute the forward operator.
# F = gd.compute_forward(coords, 1.0, 1.0, 1.0, data_coords)

# Put matter inside the cube.
density = np.zeros((coords.shape[0],))
density[:] = 1500.0
density[(
        (coords[:, 0] > 10) & (coords[:, 0] < 20)
        & (coords[:, 1] > 20) & (coords[:, 1] < 22)
        & (coords[:, 2] > 10) & (coords[:, 2] < 40))] = 2000.0

# Verify by plottting.
import volcapy.plotting.plot as plt
plt.plot(density, coords, cmin=1500.0, cmax=2000.0, n_sample=140000)

from vtk.util import numpy_support
import vtk


data = density.reshape((50, 50, 50), order="F")

# vtkImageData is the vtk image volume type
imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=True,
        array_type=vtk.VTK_DOUBLE)

# fill the vtk image data object
imdata.SetDimensions(data.shape)
imdata.SetSpacing([1,1,1])
imdata.SetOrigin([0,0,0])
imdata.GetPointData().SetScalars(depthArray)

# f.ex. save it as mhd file
writer = vtk.vtkMetaImageWriter()
writer.SetFileName("test.mhd")
writer.SetInputData(imdata)
writer.Write()
