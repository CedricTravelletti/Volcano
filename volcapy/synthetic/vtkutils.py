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
    data = data.reshape(shape, order="F")

    # vtkImageData is the vtk image volume type
    imdata = vtk.vtkImageData()
    # this is where the conversion happens
    depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=True,
            array_type=vtk.VTK_DOUBLE)
    
    # fill the vtk image data object
    imdata.SetDimensions(data.shape)
    imdata.SetSpacing([res_x,res_y,res_z])
    imdata.SetOrigin([0,0,0])
    imdata.GetPointData().SetScalars(depthArray)
    
    # f.ex. save it as mhd file
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imdata)
    writer.Write()

def ndarray_to__vtk(data, res_x, res_y, res_z, filename):
    """ Save data to vtk format.

    THIS IS THE ONE THAT WORKS WITH REAL DATA.

    Parameters
    ----------
    data: ndarray
        1D array.
    filename: string

    """
    vtk_data_array = numpy_support.numpy_to_vtk(
                num_array=data.transpose(2, 1, 0).ravel(),
                deep=True, array_type=vtk.VTK_FLOAT)

    # Convert the VTK array to vtkImageData
    imdata = vtk.vtkImageData()
    imdata.SetDimensions(data.shape)
    imdata.SetSpacing([res_x,res_y,res_z])
    imdata.SetOrigin([0,0,0])
    imdata.GetPointData().SetScalars(vtk_data_array)
    
    # f.ex. save it as mhd file
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imdata)
    writer.Write()
