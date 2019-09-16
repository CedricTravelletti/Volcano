from vtk.util import numpy_support
import vtk


def save_vtk(data, shape, res_x, res_y, res_z, filename):
    """ Save data to vtk format.

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
