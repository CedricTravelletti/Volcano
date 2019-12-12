# -*- coding: utf-8 -*-
""" Plot a point cloud using VTK.

Adapted from https://sukhbinder.wordpress.com/2013/09/17/python-vtk-script-to-display-3d-xyz-data/

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
            point_data = 0
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
 
 
if __name__ == '__main__':
    import sys
 
    if len(sys.argv) < 2:
         print('Usage: plot_point_cloud_vtk.py coords.txt')
         sys.exit()
    pointCloud = VtkPointCloud()
    pointCloud=load_data(sys.argv[1],pointCloud)
 
 
# Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
#renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()
 
# Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
 
# Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
 
# Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("Point Cloud Viewer:"+sys.argv[1])
    renderWindowInteractor.Start()
