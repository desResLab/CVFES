import numpy as np
from AortaFlowCalculate import FlowCalc

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

""" Calculate the c value based on inlet velocity function/plot.
    Then calculate the artificial compressible coefficient (inv-epsilon) and
    time step based on the c value.
"""

scale = 2.0
maxInletVelocity = 14.0
# inletfunc = '2.759e4*t**4-1.655e4*t**3+2.548e3*t**2-1.9565e1*t+1.5 if t>=0.0 and t<0.28 else -25.0*(t-0.28)+2.0 if t<=0.3 else 1.5-20.0*(t-0.3) if t<=0.35 else 0.5+5.0*(t-0.35) if t<=0.65 else -5.0*(t-0.65)+2.0 if t<=0.75 else 1.5'
lumenfile = 'Examples/CylinderProject/mesh-complete/mesh-complete.mesh.vtu'


def ReadMesh(filename):
    
    reader = vtk.vtkXMLPolyDataReader() if meshConfig.file_path.endswith('vtp') else vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    polyDataModel = reader.GetOutput()

    # Set the nodes and coordinates.
    nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
    nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
    nodes = nodes.astype(float)

    # Set the element groups, will be updated to sub-group after partition.
    nElements = polyDataModel.GetNumberOfCells()
    elementNodeIds = np.empty((nElements, 4)) # 4 nodes for tetrohedron
    for iElm in range(nElements):
        vtkCell = polyDataModel.GetCell(iElm)
        for iNode in range(4):
            elementNodeIds[iElm,iNode] = vtkCell.GetPointId(iNode)

    return nodes, elementNodeIds


def CalcInscribeDiameters(nodes, elementNodeIds):
    """ Calculate the inscribe sphere diameter of the tetrohedron elements. """
    nElements = elementNodeIds.shape[0]
    inscribeDiameters = np.empty(nElements)

    for iElm, elm in enumerate(elementNodeIds):
        # Get the edges need
        OA = nodes[elm[1]] - nodes[elm[0]]
        OB = nodes[elm[2]] - nodes[elm[0]]
        OC = nodes[elm[3]] - nodes[elm[0]]
        AB = nodes[elm[2]] - nodes[elm[1]]
        AC = nodes[elm[3]] - nodes[elm[1]]
        
        # Get the volume of the element
        volume = abs(OC[0]*(OA[1]*OB[2]-OA[2]*OB[1]) - OC[1]*(OA[0]*OB[2]-OA[2]*OB[0]) \
                + OC[2]*(OA[0]*OB[1]-OA[1]*OB[0])) / 6.0
        # Get the surface area
        area = sqrt((OA[1]*OC[2]-OA[2]*OC[1])**2 \
                    + (OA[0]*OC[2]-OA[2]*OC[0])**2 \
                    + (OA[0]*OC[1]-OA[1]*OC[0])**2) \
             + sqrt((OA[1]*OB[2]-OA[2]*OB[1])**2 \
                    + (OA[0]*OB[2]-OA[2]*OB[0])**2 \
                    + (OA[0]*OB[1]-OA[1]*OB[0])**2) \
             + sqrt((OB[1]*OC[2]-OB[2]*OC[1])**2 \
                    + (OB[0]*OC[2]-OA[2]*OC[0])**2 \
                    + (OB[0]*OC[1]-OA[1]*OC[0])**2) \
             + sqrt((AB[1]*AC[2]-AB[2]*AC[1])**2 \
                    + (AB[0]*AC[2]-AB[2]*AC[0])**2 \
                    + (AB[0]*AC[1]-AB[1]*AC[0])**2)
        area = area * 0.5

        # Calc the radius of the inscribed sphere
        r = 6.0*volume/area
        inscribeDiameters[iElm] = 2.0*r

    return inscribeDiameters


def CValue(maxInletVelocity, scale):
    ASS = 5.0
    return ASS*scale*maxInletVelocity


if __name__ == '__main__':

    c = CValue(maxInletVelocity, scale)

    inv_epsilon = c**2
    print('The artificial compressible coefficient inv-epsilon is {}'.format(inv_epsilon))

    # Calculate the minimum time step.
    nodes, elementNodeIds = ReadMesh(lumenfile)
    inscribeDiameters = CalcInscribeDiameters(nodes, elementNodeIds)
    dt = np.amin(inscribeDiameters) / c
    print('The time step is {}'.format(dt))

    # Run inlet flow calculator.
    stime = 0.0
    etime = 1.5
    cycletime = 0.75
    eqn = '2.759e4*t**4-1.655e4*t**3+2.548e3*t**2-1.9565e1*t+1.5 if t>=0.0 and t<0.28 else -25.0*(t-0.28)+2.0 if t<=0.3 else 1.5-20.0*(t-0.3) if t<=0.35 else 0.5+5.0*(t-0.35) if t<=0.65 else -5.0*(t-0.65)+2.0 if t<=0.75 else 1.5'

    flow = FlowCalc(stime, etime, dt, cycletime, eqn)
    flow[:,1] = -1000.0/60.0 * flow[:,1]
    np.savetxt('../cfg/cylinderExplicitVMSInlet.flow', flow)

    plt.plot(flow[:,0], flow[:,1])
    plt.show()

