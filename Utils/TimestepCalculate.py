import numpy as np
from math import sqrt
import os

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

# mesh_file_path = '../Examples/lc/SparseWallProperties/YoungsModulus.vtp'
# mesh_file_path = '../Examples/CylinderProject/WallProperties/YoungsModulus.vtp'
mesh_file_path = '../Examples/CylinderProject/RefineWallProperties/YoungsModulus.vtp'


def CalcTimestep(filename):
    # Read mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polyDataModel = reader.GetOutput()

    # Readin nodes.
    nNodes = polyDataModel.GetNumberOfPoints()
    nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
    nodes = nodes.astype(float)

    E = np.zeros((nNodes, 100))
    for i in range(100):
        E[:,i] = vtk_to_numpy(polyDataModel.GetPointData().GetArray('RandomField {}'.format(i+1)))

    # Loop through cells.
    nElements = polyDataModel.GetNumberOfCells()
    timestep = np.zeros(nElements)

    for iElm in range(nElements):
        vtkCell = polyDataModel.GetCell(iElm)

        nIds = np.array([vtkCell.GetPointId(ipt) for ipt in range(3)])
        perimeter = np.linalg.norm(nodes[nIds[0]]-nodes[nIds[1]]) \
                  + np.linalg.norm(nodes[nIds[0]]-nodes[nIds[2]]) \
                  + np.linalg.norm(nodes[nIds[1]]-nodes[nIds[2]])
        area = vtkCell.ComputeArea()
        diameter = 4.0 * area / perimeter

        # cE = np.amax(np.mean(E[nIds], axis=0))
        cE = np.amax(np.amax(E[nIds], axis=0))
        c = sqrt(cE / 1.0)
        timestep[iElm] = diameter / c

    return np.amin(timestep)


if __name__ == '__main__':

    filename, fileExtension = os.path.splitext(mesh_file_path)
    rhos = np.array([0.95, 3.7, 7.2])
    for rho in rhos:
        timestep = CalcTimestep('{}{}{}'.format(filename, rho, fileExtension))
        print('rho {} --- timestep {}'.format(rho, timestep))
