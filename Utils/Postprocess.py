''' Calculate the 'shear' stress along the wall for coronary model.
'''

import numpy as np
import xml.etree.ElementTree as ET
# Calculate the global stress tensor
from optimizedSolidStressCalculate import OptimizedCalculateStress
from optimizedSolidStressCalculate import TranformStress

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk


def parseXML(xmlfile):

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    newitems = []
    for item in root.findall('./timestep/path_element/path_points/path_point'):
        # print(item.get('id'))
        for subitem in item:
            if subitem.tag == 'pos':
                newitems.append([float(subitem.get('x')), float(subitem.get('y')), float(subitem.get('z'))])

    return np.array(newitems)


class Model:
    def __init__(self, nNodes, nodes, nElements, elements, u, up, E):

        self.nNodes = nNodes
        self.nodes = nodes
        self.nElements = nElements
        self.elements = elements
        self.u = u
        self.up = up
        self.E = E

        self.glbStress = None

def readinModel(dispfile, Efile, nSmp):

    # Readin the vtk file.
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(dispfile)
    reader.Update()
    polyDataModel = reader.GetOutput()

    # Readin nodes.
    nNodes = polyDataModel.GetNumberOfPoints()
    nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
    nodes = nodes.astype(float)

    # Loop through cells.
    nElements = polyDataModel.GetNumberOfCells()
    elementNodeIds = np.zeros((nElements, 3), dtype=int)
    for iElm in range(nElements):
        vtkCell = polyDataModel.GetCell(iElm)
        elementNodeIds[iElm,:] = np.array([vtkCell.GetPointId(ipt) for ipt in range(3)])

    # Readin the displacement u and up.
    u = np.zeros((3*nNodes, nSmp))
    up = np.zeros((3*nNodes, nSmp))
    for iSmp in range(nSmp):
        # u[:,iSmp] = vtk_to_numpy(polyDataModel.GetPointData().GetArray('{}_{:03d}'.format('displacement', iSmp))).ravel()
        u[:,iSmp] = vtk_to_numpy(polyDataModel.GetPointData().GetArray('{}_{:03d}'.format('u', iSmp))).ravel()
        # up[:,iSmp] = vtk_to_numpy(polyDataModel.GetPointData().GetArray('{}_{:03d}'.format('up', iSmp))).ravel()


    # Readin the Young's Modulus.
    # reader = vtk.vtkXMLPolyDataReader()
    # reader.SetFileName(Efile)
    # reader.Update()
    # polyDataModel = reader.GetOutput()

    # E = np.zeros((nNodes, nSmp))
    # for iSmp in range(nSmp):
    #     E[:,iSmp] = vtk_to_numpy(polyDataModel.GetPointData().GetArray('RandomField {}'.format(iSmp+1)))
    E = np.load(Efile)

    return Model(nNodes, nodes, nElements, elementNodeIds, u, up, E)


def calcGlbStress(dispfile, Efile, nSmp):
    ''' Calculate the global stress tensor.
        u: (ndof, nSmp)
    '''

    model = readinModel(dispfile, Efile, nSmp)

    # Young's Modulus
    elmAvgE = np.mean(model.E[model.elements,:], axis=1)
    # D
    k = 5.0/6.0
    v = 0.4
    D = np.array([[1.0,   v,       0.0,         0.0,         0.0],
                  [  v, 1.0,       0.0,         0.0,         0.0],
                  [0.0, 0.0, 0.5*(1-v),         0.0,         0.0],
                  [0.0, 0.0,       0.0, 0.5*k*(1-v),         0.0],
                  [0.0, 0.0,       0.0,         0.0, 0.5*k*(1-v)]])/(1-v*v)


    model.glbStress = np.empty((model.nElements, nSmp, 3, 3))
    updateNodes = np.tile(model.nodes, (nSmp, 1, 1))
    # updateNodes = updateNodes.reshape(nSmp, 3*model.nNodes) + model.up.transpose()
    updateNodes = updateNodes.reshape(nSmp, 3*model.nNodes)
    model.updateNodes = updateNodes.reshape(nSmp, model.nNodes, 3)
    OptimizedCalculateStress(model.updateNodes, model.elements,
                             D, elmAvgE, model.u, model.glbStress)

    return model


def getControlPoints(model, paths):

    elmNodes = model.nodes[model.elements, :]
    elmCenters = np.mean(elmNodes, axis=1)

    elmCtrlPts = np.zeros((model.nElements, 2, 3))
    for iElm in range(model.nElements):
        distances = np.array([np.linalg.norm(path - elmCenters[iElm], axis=1) for path in paths])

        minpath = 0
        minidx = 0
        minvalue = 10000.0
        for ipath in range(len(paths)):
            idx = np.argmin(distances[ipath])
            value = distances[ipath][idx]
            if value < minvalue:
                minpath, minidx, minvalue = ipath, idx, value

        elmCtrlPts[iElm,:,:] = paths[minpath][[minidx, minidx-1],:] if minidx > 0 else paths[minpath][[minidx+1, minidx],:]

    return elmCtrlPts


def transformStress(model, paths, nSmp):

    elmCtrlPts = getControlPoints(model, paths)

    # for debugging
    # model.elmZ = np.empty((model.nElements, 3))
    model.tStress = np.empty((model.nElements, nSmp, 3, 3))
    model.tU = np.zeros((model.nNodes*3, nSmp))
    TranformStress(model.updateNodes, model.elements, elmCtrlPts,
                   model.glbStress, model.tStress, model.u, model.tU)
                   # model.glbStress, model.tStress, model.elmZ)


def writeStress(model, modelfile, resultfile, nSmp):

    # Read mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(modelfile)
    reader.Update()
    polyDataModel = reader.GetOutput()

    # Add result and write
    dim = np.array(['xx', 'xy', 'xz', 'yy', 'yz', 'zz'])
    idx = np.array([[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]])
    for i, name in enumerate(dim):
        for iSmp in range(nSmp):
            stressVec = numpy_to_vtk(model.tStress[:,iSmp,idx[i,0], idx[i,1]])
            stressVec.SetName('{}_{:03d}'.format(name, iSmp))
            polyDataModel.GetCellData().AddArray(stressVec)

    for iSmp in range(nSmp):
        uVec = numpy_to_vtk(model.tU[:,iSmp].reshape((model.nNodes, 3)))
        uVec.SetName('tDisplacement_{:03d}'.format(iSmp))
        polyDataModel.GetPointData().AddArray(uVec)

    # # for debugging
    # elmZVec = numpy_to_vtk(model.elmZ)
    # elmZVec.SetName('localZ')
    # polyDataModel.GetCellData().AddArray(elmZVec)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polyDataModel)
    writer.SetFileName(resultfile)
    writer.Write()

    print('Write result to {}.'.format(resultfile))


def main(paths, modelfile, dispfile, Efile, resultfile, nSmp):

    model = calcGlbStress(dispfile, Efile, nSmp)

    # Transform to "Shear" stress along the wall.
    transformStress(model, paths, nSmp)

    # Write to file.
    writeStress(model, modelfile, resultfile, nSmp)


def tellDifference(nElms, nSmp):
    ''' Tell the difference btw stress calculated using
        updated and non-updated nodes coordinates.
    '''
    file1 = '../Examples/CylinderProject/SparseResultRho0.95/Tstress_2cycles37500.vtp'
    file2 = '../Examples/CylinderProject/SparseResultRho0.95/NTstress_2cycles37500.vtp'
    files = [file1, file2]

    dim = np.array(['xx', 'xy', 'xz', 'yy', 'yz', 'zz'])
    stresses = np.empty((len(files), len(dim), nSmp, nElms))

    for ifile in range(len(files)):
        # Read mesh
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(files[ifile])
        reader.Update()
        polyDataModel = reader.GetOutput()

        # Add result and write
        for i, name in enumerate(dim):
            for iSmp in range(nSmp):
                stresses[ifile, i, iSmp, :] = vtk_to_numpy(polyDataModel.GetCellData().GetArray('{}_{:03d}'.format(name, iSmp)))

    print('The difference percentage is {}'.format(np.linalg.norm(stresses[1]-stresses[0])/np.linalg.norm(stresses[0])))


# if __name__ == "__main__":

#     # Get the points along the centerlines.
#     pathfiles = ['../Examples/lc/CenterLinePath/lc1.pth',
#                  '../Examples/lc/CenterLinePath/lc1_sub1.pth',
#                  '../Examples/lc/CenterLinePath/lc1_sub2.pth',
#                  '../Examples/lc/CenterLinePath/lc1_sub3.pth',
#                  '../Examples/lc/CenterLinePath/lc2.pth',
#                  '../Examples/lc/CenterLinePath/lc2_sub1.pth']

#     paths = []
#     for pathfile in pathfiles:
#         paths.append(parseXML(pathfile))

#     # Calc global stress tensor.
#     # modelfile = '../Examples/lc/mesh-complete-5layers/walls_combined.vtp'
#     # Efile = '../Examples/lc/lc5LayersWallProperties/lcYoungsModulus0.95.npy'

#     # nTimesteps = np.arange(5000, 800000, 5000, dtype=int)
#     # nTimesteps = np.append(nTimesteps, 800001)

#     # for i in nTimesteps:
#     #     # dispfile = '../Examples/lc/PaperResults/PulResults5LayersLcInflow095/displacement_lcSparse{}.vtp'.format(i)
#     #     # resultfile = '../Examples/lc/PaperResults/PulResults5LayersLcInflow095/stress_{}.vtp'.format(i)

#     #     dispfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lc5LayersPulsatileResults/Results095/displacement_{}.vtp'.format(i)
#     #     resultfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lc5LayersPulsatileResults/Results095/stress_{}.vtp'.format(i)

#     #     main(paths, modelfile, dispfile, Efile, resultfile, 100)


#     # dispfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lc5LayersPulsatileResults/Results72/displacement_800001.vtp'
#     # resultfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lc5LayersPulsatileResults/Results72/stress_800001.vtp'
#     # main(paths, modelfile, dispfile, Efile, resultfile, 100)


#     modelfile = '../Examples/lc/lcSparse-mesh-complete/walls_combined.vtp'
#     Efile = '../Examples/lc/SparseWallProperties/YoungsModulus0.95.npy'
#     dispfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lcSparseSteadyState/Results72/displacement_lcSparse560000.vtp'
#     resultfile = '/media/xli/PHDWORK_XLI/2020PaperResults/lcSparseSteadyState/Results72/stress_lcSparse560000.vtp'
#     main(paths, modelfile, dispfile, Efile, resultfile, 100)


#     # modelfile = '../Examples/CylinderProject/mesh-complete/walls_combined.vtp'
#     # dispfile = '../Examples/CylinderProject/SparseResultRho0.95/WithoutStress/displacement_2cycles37500.vtp'
#     # Efile = '../Examples/CylinderProject/WallProperties/YoungsModulus0.95.npy'
#     # resultfile = '../Examples/CylinderProject/SparseResultRho0.95/stress_2cycles37500.vtp'

#     # tellDifference(5074, 100)


if __name__ == '__main__':
    
    paths = [np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 23.0]]),
             np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 13.0]]),
             np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])]

    # Calc global stress tensor.
    modelfile = '../Examples/CylinderProject/refine-more-mesh-complete/walls_combined.vtp'
    Efile = '../Examples/CylinderProject/MoreRefineWallProperties/cyYoungsModulus0.95.npy'

    # nTimesteps = np.arange(1000, 150001, 1000, dtype=int)
    nTimesteps = np.arange(1000, 50001, 1000, dtype=int)

    for i in nTimesteps:
        # dispfile = '../Examples/lc/PaperResults/PulResults5LayersLcInflow095/displacement_lcSparse{}.vtp'.format(i)
        # resultfile = '../Examples/lc/PaperResults/PulResults5LayersLcInflow095/stress_{}.vtp'.format(i)

        dispfile = '/media/xli/PHDWORK_XLI/2020PaperResults/cyPulsatileResults/cyPulResults095/displacement_cySS{}.vtp'.format(i)
        resultfile = '/media/xli/PHDWORK_XLI/2020PaperResults/cyPulsatileResults/cyPulResults095/stress_cySS{}.vtp'.format(i)
        main(paths, modelfile, dispfile, Efile, resultfile, 100)

