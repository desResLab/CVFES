# Export the body stress from the fluid result
# calculated by SimVascular.
import numpy as np
from bdyStressExport import BdyStressExport

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

# 1. Read in the fluid result containing velocity and pressure.
# 2. Read in the fluid file for geometry.
# 3. Read in the solid file for geometry.
# 4. Find the geometry connection.
# 5. Call the cython function to export the body stress.

class Fluid:

    def __init__(self, fluidFile, vName, pName):
        
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fluidFile)
        reader.Update()

        polyDataModel = reader.GetOutput()

        self.nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
        self.nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
        self.nodes = self.nodes.astype(float)
        self.glbNodeIds = vtk_to_numpy(polyDataModel.GetPointData().GetArray('GlobalNodeID'))
        self.glbElementIds = vtk_to_numpy(polyDataModel.GetCellData().GetArray('GlobalElementID'))

        # Set the element groups, will be updated to sub-group after partition.
        self.nElements = polyDataModel.GetNumberOfCells()
        elements = np.empty((self.nElements, 4), dtype=np.int64)
        for iElm in range(self.nElements):
            vtkCell = polyDataModel.GetCell(iElm)
            elements[iElm] = np.array([vtkCell.GetPointId(ipt) for ipt in range(4)])

        # Pullout the nodes of each element to construct element's nodes matrix.
        self.elementNodeIds = elements

        # Read the velocity and pressure result.
        self.du = vtk_to_numpy(polyDataModel.GetPointData().GetArray(vName))
        self.p = vtk_to_numpy(polyDataModel.GetPointData().GetArray(pName))

class Solid:
    """docstring for solid"""
    def __init__(self, solidFile):
        
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(solidFile)
        reader.Update()

        polyDataModel = reader.GetOutput()

        self.nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
        self.nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
        self.nodes = self.nodes.astype(float)
        self.glbNodeIds = vtk_to_numpy(polyDataModel.GetPointData().GetArray('GlobalNodeID'))
        self.glbElementIds = vtk_to_numpy(polyDataModel.GetCellData().GetArray('GlobalElementID'))

        # Set the element groups, will be updated to sub-group after partition.
        self.nElements = polyDataModel.GetNumberOfCells()
        elements = np.empty((self.nElements, 3), dtype=np.int64)
        for iElm in range(self.nElements):
            vtkCell = polyDataModel.GetCell(iElm)
            elements[iElm] = np.array([vtkCell.GetPointId(ipt) for ipt in range(3)])

        # Pullout the nodes of each element to construct element's nodes matrix.
        self.elementNodeIds = elements
        self.polyDataModel = polyDataModel

def SaveVtp(val, uname, polyDataModel, filename):

    uTuples = numpy_to_vtk(val)
    uTuples.SetName(uname)
    polyDataModel.GetPointData().AddArray(uTuples)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polyDataModel)
    writer.SetFileName(filename)
    writer.Write()

    print('Write result to {}.'.format(filename))


def main(solidFile, fluidFile, vName, pName, exportFilename):

    lumen = Fluid(fluidFile, vName, pName)
    wall = Solid(solidFile)
    
    # Identify wall nodes on lumen border.
    sorter = np.argsort(lumen.glbNodeIds)
    lumenWallNodeIds = sorter[np.searchsorted(lumen.glbNodeIds, wall.glbNodeIds, sorter=sorter)]
    
    # Identify elements attached on the wall.
    sorter = np.argsort(lumen.glbElementIds)
    elementIds = sorter[np.searchsorted(lumen.glbElementIds, wall.glbElementIds, sorter=sorter)]
    lumenWallElements = lumen.elementNodeIds[elementIds]

    lDN = np.array([[-1.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0]])

    wallStress = np.zeros((wall.nNodes, 3), dtype=np.float)
    BdyStressExport(lumen.nodes, lumenWallElements, lumenWallNodeIds,
                    wall.nodes, wall.elementNodeIds,
                    lumen.du, lumen.p, lDN, wallStress)

    np.save(exportFilename, wallStress)

    # Write to the vtp files.
    SaveVtp(wallStress, 'f', wall.polyDataModel, '{}.vtp'.format(exportFilename))


if __name__ == '__main__':

    # solidFile = 'Examples/CylinderProject/refine-more-mesh-complete-732470/walls_combined.vtp'
    # fluidFile = 'Examples/CylinderProject/MoreFineResults/solution_01500.vtu'
    # vName = 'velocity'
    # pName = 'pressure'
    # exportFilename = 'Examples/CylinderProject/MoreFineResults/cySS_wallStress'
    # main(solidFile, fluidFile, vName, pName, exportFilename)


    # solidFile = 'Examples/CylinderProject/refine-more-mesh-complete-732470/walls_combined.vtp'
    # fluidFile = 'Examples/CylinderProject/MoreFineResults/solution_'
    # vName = 'velocity'
    # pName = 'pressure'
    # exportFilename = 'Examples/CylinderProject/MoreFineResults/cySS_wallStress_'
    # expIndex = 0
    # for i in range(2700, 3001, 10):
    #     ffile = '{}{:05d}.vtu'.format(fluidFile, i)
    #     expfile = '{}{}'.format(exportFilename, expIndex)
    #     expIndex += 1
    #     main(solidFile, ffile, vName, pName, expfile)

    solidFile = 'Examples/lc/lcSparse-mesh-complete/walls_combined.vtp'
    fluidFile = 'Examples/lc/ResultsFourier40/solution_'
    vName = 'velocity'
    pName = 'pressure'
    exportFilename = 'Examples/lc/ResultsFourier40/lcPuls_wallStress_'
    expIndex = 0
    for i in range(7000, 8401, 10):
        ffile = '{}{:05d}.vtu'.format(fluidFile, i)
        expfile = '{}{}'.format(exportFilename, expIndex)
        expIndex += 1
        main(solidFile, ffile, vName, pName, expfile)

