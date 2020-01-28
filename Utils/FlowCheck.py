import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

from mesh import Face

def ProcessFaces(faces, nodes):
    for face in faces:
        # Calculate the inlet area used for calculating BC velocity.
        face.area = np.sum(face.elementAreas)

        # Calculate the unit norm vector of this inlet.
        nodeIds = face.glbNodeIds
        elmNIds = face.elementNodeIds
        v = np.array([nodes[nodeIds[elmNIds[0,1]]] - nodes[nodeIds[elmNIds[0,0]]],
                      nodes[nodeIds[elmNIds[0,2]]] - nodes[nodeIds[elmNIds[0,0]]]])
        elmNormV = np.cross(v[0], v[1])
        face.normal = elmNormV / np.linalg.norm(elmNormV)

def ReadResult(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    polyDataModel = reader.GetOutput()
    pressure = vtk_to_numpy(polyDataModel.GetPointData().GetArray('pressure'))
    velocity = vtk_to_numpy(polyDataModel.GetPointData().GetArray('velocity'))
    return pressure, velocity

mesh_file_path = 'Examples/lc/lcSparse-mesh-complete/mesh-complete.mesh.vtu'
inlet_file_path = 'Examples/lc/lcSparse-mesh-complete/mesh-surfaces/inlet.vtp'
outlet_file_path = 'Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_1.vtp','Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_2.vtp','Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_3.vtp','Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_4.vtp','Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_5.vtp','Examples/lc/lcSparse-mesh-complete/mesh-surfaces/outlet_6.vtp'
# result_file_path = 'Examples/lc/SparseResults/sparse_stress_pulse'
result_file_path = 'sparse_stress_pulse'

# Read mesh
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file_path)
reader.Update()
polyDataModel = reader.GetOutput()
nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
nodes = nodes.astype(float)
glbNodeIds = vtk_to_numpy(polyDataModel.GetPointData().GetArray('GlobalNodeID'))

faces = []
faces.append(Face(inlet_file_path, glbNodeIds))
for f in outlet_file_path:
    faces.append(Face(f, glbNodeIds))
ProcessFaces(faces, nodes)

# Calculate flow and average pressure
# dt = 0.001
# T = 0.7
# N = int(T/dt)
# dn = 10

dt = 0.001
T = 0.001
N = int(T/dt)
dn = 1

pressureTlb = np.empty((len(faces)+1, int(N/dn)+1))
pressureTlb[0,:] = np.arange(dn, N+2, dn)*dt
flowTlb = np.empty((len(faces)+1, int(N/dn)+1))
flowTlb[0,:] = np.arange(dn, N+2, dn)*dt

# Loop through result at each time step
for i in range(int(N/dn)+1):
    # Read result file
    resfile = '{}{}.vtu'.format(result_file_path, (i+1)*dn)
    print(resfile)
    p, v = ReadResult(resfile)
    # Loop through each face
    for iface, face in enumerate(faces):
        nodeIds = face.glbNodeIds
        # Calc flow (u x area), and avg pressure
        flow = 0.0
        force = 0.0
        for ielm, ids in enumerate(face.elementNodeIds):
            ids = nodeIds[ids]
            flow += np.dot(np.sum(v[ids], axis=0)/3.0, face.normal)*face.elementAreas[ielm]
            force += np.sum(p[ids])/3.0*face.elementAreas[ielm]
        # Set the table
        flowTlb[iface+1, i] = flow
        pressureTlb[iface+1, i] = force/face.area

np.save('Flow', flowTlb)
np.save('Pressure', pressureTlb)

