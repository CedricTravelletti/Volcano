# -*- coding: utf-8 -*-
""" Plot volcano as point cloud together with measuremnt sites.

"""
import vtk
from numpy import random,genfromtxt,size
 
     
class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
 
    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            point_data = point[2]
            self.vtkDepth.InsertNextValue(point_data)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)

            # Perso
            self.vtkActor.GetProperty().SetPointSize(2)
        """
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        """
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
        
        # Perso
        self.vtkActor.Modified()

    def addPointData(self, point):
        """ Modified function so we set a different color to the datapoints.

        """
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            point_data = 10
            self.vtkDepth.InsertNextValue(point_data)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)

            # Perso
            self.vtkActor.GetProperty().SetPointSize(5)
        """
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        """
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
        
        # Perso
        self.vtkActor.Modified()
 
    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
 
def load_data(filename,pointCloud):
    data = genfromtxt(filename,dtype=float,skip_header=2,usecols=[0,1,2])
     
    for k in range(size(data,0)):
        point = data[k] #20*(random.rand(3)-0.5)
        pointCloud.addPoint(point)
         
    return pointCloud

def load_data_data(filename,pointCloud):
    """ Modified function so we set a different color to the datapoints.

    """
    data = genfromtxt(filename,dtype=float,skip_header=2,usecols=[0,1,2])
     
    for k in range(size(data,0)):
        point = data[k] #20*(random.rand(3)-0.5)
        pointCloud.addPointData(point)
         
    return pointCloud
 
 
if __name__ == '__main__':
    import sys
 
    if len(sys.argv) < 2:
         print('Usage: plot_measurements.py volcano_coords.txt data_coords.txt')
         sys.exit()
    pointCloud_coords = VtkPointCloud(zMin=-500.0, zMax=1000.0)
    pointCloud_coords=load_data(sys.argv[1],pointCloud_coords)

    pointCloud_data = VtkPointCloud()
    pointCloud_data=load_data_data(sys.argv[2],pointCloud_data)
 
 
# Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud_coords.vtkActor)
    renderer.AddActor(pointCloud_data.vtkActor)
#renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()
 
# Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
 
# Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    # If want behaviour similar to ParaView.
    # renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
 
# Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer:"+sys.argv[1])
    renderWindowInteractor.Start()
