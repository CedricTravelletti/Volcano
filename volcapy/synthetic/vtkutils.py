""" Utilities to convert inversion data to VTK format for 3d visualization.

"""
from vtk.util import numpy_support
import vtk
import numpy as np


def save_vtk(data, shape, res_x, res_y, res_z, filename):
    """ Save data to vtk format.

    THIS ONLY WORKS FOR SYNTHETIC DATA. REAL DATA HAS TO BE TRANSPOSED.

    Parameters
    ----------
    data: ndarray
        1D array.
    shape: (int, int, int)
    filename: string

    """
    # Consider 0s and below as NaNs. This makes visualization easier
    # using the *Threshold* filter in Paraview.
    data[data<=0.0] = np.nan

    # Checks
    print("Data shape {}".format(data.shape))
    print("Target shape {}".format(shape))
    print("Target shape (total) {}".format(shape[0]*shape[1]*shape[2]))

    # data = data.reshape(shape, order="F")
    # VTK uses strange ordering (z first) so have to transpose data.
    data = data.reshape(shape, order="C") # Back to 3d.

    # vtkImageData is the vtk image volume type
    imdata = vtk.vtkImageData()
    # This is where the conversion happens.
    # Pay attention, VTK uses Fortran-like ordering, where first index loops
    # fastest.
    depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True,
            array_type=vtk.VTK_DOUBLE)
    
    # fill the vtk image data object
    imdata.SetDimensions(shape)
    imdata.SetSpacing([res_x,res_y,res_z])
    imdata.SetOrigin([0,0,0])
    imdata.GetPointData().SetScalars(depthArray)
    
    # f.ex. save it as mhd file
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imdata)
    writer.Write()

def ndarray_to_vtk(data, res_x, res_y, res_z, filename):
    """ Save data to vtk format.

    THIS IS THE ONE THAT WORKS WITH REAL DATA.

    Parameters
    ----------
    data: ndarray
        1D array.
    filename: string

    """
    # Consider 0s and below as NaNs. This makes visualization easier
    # using the *Threshold* filter in Paraview.
    data[data<=0.0] = np.nan

    # See the TRANSPOSE? VTK uses strange ordering.
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(2, 1, 0).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    """
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(0, 2, 1).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(1, 2, 0).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(2, 0, 1).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(1, 0, 2).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(0, 1, 2).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)
    """

    # Convert the VTK array to vtkImageData
    imdata = vtk.vtkImageData()
    imdata.SetDimensions(shape)
    imdata.SetSpacing([res_x,res_y,res_z])
    imdata.SetOrigin([0,0,0])
    imdata.GetPointData().SetScalars(vtk_data_array)
    
    # f.ex. save it as mhd file
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imdata)
    writer.Write()
