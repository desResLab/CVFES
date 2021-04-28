#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVFES class is the main solver of the CVFES project.
"""
from cvconfig import CVConfig
# from GMRF import GMRF
import numpy as np
import os.path
# from os.path import splitext
import sys
from math import pi
from math import sqrt
from parabolicVelocityProfile import ParabolicVelocityProfile

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class Element:

    def __init__(self, vtkCell):

        self.nNodes = vtkCell.GetNumberOfPoints()
        self.nodes = np.zeros(self.nNodes, dtype=np.int64)
        for ipt in range(self.nNodes):
            self.nodes[ipt] = vtkCell.GetPointId(ipt)

        # TODO::Add edges info if needed later.

class ElementOfFace(Element):

    def __init__(self, vtkCell):

        Element.__init__(self, vtkCell)

        self.area = vtkCell.ComputeArea()


class Face:

    def __init__(self, faceFilePath, tglbNodeIds):

        # self.glbNodeIds = np.empty(0)
        # self.elements = np.empty(0)
        # self.nElements = 0
        # self.appNodes = None

        self.readinMesh(faceFilePath, tglbNodeIds)

        self.elementNodeIds = np.array([elm.nodes for elm in self.elements])
        self.elementAreas = np.array([elm.area for elm in self.elements])

        # # Calculate inlet surface area.
        # if faceConfig.name == 'inlet':
        #     # mass = vtk.vtkMassProperties()
        #     # mass.SetInputData(polyDataModel)
        #     # mass.Update()
        #     # self.inletArea = mass.GetSurfaceArea()

        #     self.nElements = polyDataModel.GetNumberOfCells()
        #     self.elements = np.array([ElementOfFace(polyDataModel.GetCell(i)) for i in range(self.nElements)])
        #     self.appNodes = None

    def readinMesh(self, file_path, tglbNodeIds):
        reader = vtk.vtkXMLPolyDataReader() if file_path.endswith('vtp') else vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()

        polyDataModel = reader.GetOutput()
        # Find the indices of the face nodes.
        glbNodeIds = vtk_to_numpy(polyDataModel.GetPointData().GetArray('GlobalNodeID'))
        # self.glbNodeIds -= 1
        # self.glbNodeIds = np.where(np.in1d(tglbNodeIds, glbNodeIds))[0]
        self.glbNodeIds = self.adjustNodeIds(glbNodeIds, tglbNodeIds)
        # Add in elements info, only used by 'inlet' face now.
        self.nElements = polyDataModel.GetNumberOfCells()
        self.elements = np.array([ElementOfFace(polyDataModel.GetCell(i)) for i in range(self.nElements)])

    def adjustNodeIds(self, glbNodeIds, tglbNodeIds):
        # nodeIds = np.array([np.where(tglbNodeIds==i)[0] for i in glbNodeIds]).reshape(glbNodeIds.shape)
        nodeIds = []
        for i in glbNodeIds:
            crrIdx = np.where(tglbNodeIds == i)[0]
            if len(crrIdx) > 0:
                nodeIds.append(crrIdx[0])
        return np.array(nodeIds)


class Mesh:
    """ Mesh structure, contains nodes and elements primarily.
        Right now one vtp file corresponds to one mesh and one domain.
    """

    def __init__(self, comm, config, meshConfig, eqnConfig):

        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        self.readMesh(config, meshConfig)

        # # Debug, check how many elements contain node 646 (59734)
        # if meshConfig.name == 'wall':
        #     for i in range(self.nElements):
        #         if 59734 in self.elements[i].nodes:
        #             print('Element {} contain it!'.format(i))
        #     sys.exit(0)

        self.faces = {f.name: self.readFaces(f) for f in meshConfig.faces}
        self.processFaces()

        # ------------- Related to Partition -----------------
        self.commNodeIds = None # common nodes processor contains
        self.totalCommNodeIds = None
        self.elememtsIds = None # after partition
        self.partition = None # The whole partition result, only used by root.

        # Set the initial conditions.
        self.setInitialConditions(eqnConfig.initialConditions)
        self.setBoundaryCondtions(eqnConfig.boundaryConditions)

    def readMesh(self, config, meshConfig):

        reader = vtk.vtkXMLPolyDataReader() if meshConfig.file_path.endswith('vtp') else vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(meshConfig.file_path)
        reader.Update()

        polyDataModel = reader.GetOutput()

        # # Transfer the unstructuredGridData to polyDataModel.
        # surface = vtk.vtkDataSetSurfaceFilter()
        # surface.SetInputData(polyDataModel)
        # surface.Update()
        # polyDataModel = surface.GetOutput()

        self.polyDataModel = polyDataModel

        # Set the nodes and coordinates.
        self.nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
        self.nodes = np.copy(vtk_to_numpy(polyDataModel.GetPoints().GetData())) # _nodes
        self.nodes = self.nodes.astype(float)
        self.glbNodeIds = vtk_to_numpy(polyDataModel.GetPointData().GetArray('GlobalNodeID'))
        self.glbElementIds = vtk_to_numpy(polyDataModel.GetCellData().GetArray('GlobalElementID'))

        # Set the element groups, will be updated to sub-group after partition.
        self.nElements = polyDataModel.GetNumberOfCells()
        self.elements = np.array([Element(polyDataModel.GetCell(i)) for i in range(self.nElements)])
        # Pullout the nodes of each element to construct element's nodes matrix.
        self.elementNodeIds = np.array([elm.nodes for elm in self.elements])
        # Set total number of elements in the mesh.
        self.gnElements = self.nElements

        # # Debugging !!!!!!!!
        # cell = polyDataModel.GetCell(101)
        # # for i in range(4):
        # #     print('cell point id: {}'.format(cell.GetPointId(i)))
        # cellPointIds = vtk_to_numpy(cell.GetPointIds())
        # print cellPointIds

        # Set the total number of degree of freedoms.
        self.dof = config.ndim
        self.ndof = config.ndim * self.nNodes

        # Set the domain id.
        self.domainId = meshConfig.domainId

    def readFaces(self, faceConfig):

        faces = []
        if isinstance(faceConfig.file_path, list):
            for path in faceConfig.file_path:
                faces.append(Face(path, self.glbNodeIds))
        else:
            faces.append(Face(faceConfig.file_path, self.glbNodeIds))

        return faces

    def processFaces(self):
        pass

    def setCondition(self, value, fieldname, n=None, dof=None, valueDof=None):
        """ Setting the initial conditions of the mesh based on the configuration,
            if it's presetting uniform value then set it to the whole geometry
            otherwise read it from geometry (vtp) files.
        """
        if isinstance(value, str):

            reader = vtk.vtkXMLPolyDataReader() if value.endswith('vtp') else vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(value)
            reader.Update()

            polyDataModel = reader.GetOutput()

            # TODO:: check the compatibility between initial condition file and the mesh's geometry

            prop = vtk_to_numpy(polyDataModel.GetPointData().GetArray(fieldname))

        else:

            if n is None:
                n = self.nNodes
            if dof is None:
                dof = self.dof

            prop = np.zeros((n, dof))
            if valueDof is None:
                prop[:,:] = value
            else:
                prop[:,valueDof] = value

        prop = prop.ravel()
        return prop

    def setInitialConditions(self, iniCondConfig):
        pass

    def setBoundaryCondtions(self, bdyCondConfig):
        pass


class FluidMesh(Mesh):

    def __init__(self, comm, config, meshConfig, eqnConfig):
        Mesh.__init__(self, comm, config, meshConfig, eqnConfig)

        # elementIds = np.zeros((self.nElements, 4), dtype=int)
        # elementIds[:,:] = self.elementNodeIds[:,[1,0,2,3]]
        # self.elementNodeIds = elementIds

        # ------------- Related to Calculation -----------------
        # Set the physical parameters.
        self.density = eqnConfig.density
        self.dviscosity = eqnConfig.dviscosity
        self.f = eqnConfig.f

    def processFaces(self):
        nodes = self.nodes
        
        # Collect all the inlet and outlet
        self.wall = np.array([w.glbNodeIds for w in self.faces['wall']]).ravel()

        for inletFace in self.faces['inlet']:
            elmNIds = inletFace.elementNodeIds
            inlet = inletFace.glbNodeIds
            
            inletFace.bdyFlags = np.isin(inlet, self.wall)
            # intersectNodes = inlet[flags]
            inletFace.appNodes = inlet[~inletFace.bdyFlags]

            # Calculate the inlet area used for calculating BC velocity.
            inletArea = 0.0
            for iElm, elm in enumerate(inletFace.elements):
                # numIncluding = np.sum(~np.isin(inlet[elm.nodes], intersectNodes))
                # inletArea += elm.area * numIncluding / 3.0
                inletArea += elm.area
            inletFace.inletArea = inletArea

            # Calculate the inlet paraboic velocity profile. (Prepare)
            inletBdy = np.arange(len(inlet))[inletFace.bdyFlags]
            inletFace.u, inletFace.int_u = ParabolicVelocityProfile(elmNIds, nodes, inlet, inletBdy)

            # Calculate the unit norm vector of this inlet.
            v = np.array([nodes[inlet[elmNIds[0,1]]] - nodes[inlet[elmNIds[0,0]]],
                          nodes[inlet[elmNIds[0,2]]] - nodes[inlet[elmNIds[0,0]]]])
            # elmNormV = np.cross(v[0], v[1])
            elmNormV = np.array([v[0,1]*v[1,2]-v[0,2]*v[1,1],
                                 -v[0,0]*v[1,2]+v[0,2]*v[1,0],
                                 v[0,0]*v[1,1]-v[0,1]*v[1,0]])
            inletFace.normal = elmNormV / np.linalg.norm(elmNormV)

        # Set the inlet to be appliable inlet glbNodeIds
        self.inlet = np.array([il.appNodes for il in self.faces['inlet']]).ravel()

        for outletFace in self.faces['outlet']:
            elmNIds = outletFace.elementNodeIds
            outlet = outletFace.glbNodeIds
            outletFace.bdyFlags = np.isin(outlet, self.wall)
            outletFace.appNodes = outlet[~outletFace.bdyFlags]

            # Calculate the outward normal to this outlet.
            v = np.array([nodes[outlet[elmNIds[0,1]]] - nodes[outlet[elmNIds[0,0]]],
                          nodes[outlet[elmNIds[0,2]]] - nodes[outlet[elmNIds[0,0]]]])
            elmNormV = np.array([v[0,1]*v[1,2]-v[0,2]*v[1,1],
                                 -v[0,0]*v[1,2]+v[0,2]*v[1,0],
                                 v[0,0]*v[1,1]-v[0,1]*v[1,0]])
            outletFace.normal = elmNormV / np.linalg.norm(elmNormV)

        # Set the outlet to be appliable outlet glbNodeIds
        self.outlet = np.array([ol.appNodes for ol in self.faces['outlet']]).ravel()


    def setInitialConditions(self, iniCondConfig):
        # Set the acceleration
        self.iniDDu = self.setCondition(iniCondConfig.acceleration, 'acceleration')
        # Set the velocity
        self.iniDu = self.setCondition(iniCondConfig.velocity, 'velocity')
        # Set the pressure
        self.iniP = self.setCondition(iniCondConfig.pressure, 'pressure', dof=1)

    def setBoundaryCondtions(self, bdyCondConfig):
        # Set the inlet velocity BC.
        self.parabolicInlet = bdyCondConfig.parabolicInlet

        if isinstance(bdyCondConfig.inletVelocity, str):
            if bdyCondConfig.inletVelocity.endswith('.flow'):
                self.inletVelocity = np.loadtxt(bdyCondConfig.inletVelocity)
            else:
                self.inletVelocity = bdyCondConfig.inletVelocity
                # volumeVelocity = eval(self.inletVelocity, {'t':0.0})
        else:
            self.inletVelocity = None
            # Set up the constant inlet velocity used for all time steps.
            volumeVelocity = bdyCondConfig.inletVelocity
            for inletFace in self.faces['inlet']:
                self.calcInletVelocity(volumeVelocity, inletFace)

        # Set the outlet Natural BC.
        # TODO:: Find out how to Apply Natrual BC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for outletFace in self.faces['outlet']:
            nOutlet = len(outletFace.glbNodeIds)
            outletFace.ouletH = self.setCondition(bdyCondConfig.outletH, 'h', n=nOutlet, dof=1)

    def updateInletVelocity(self, t):
        if self.inletVelocity is None:
            return

        if isinstance(self.inletVelocity, np.ndarray):
            volumeVelocity = self.inletVelocity[self.inletVelocity[:,0]==t, 1]
        else:
            volumeVelocity = eval(self.inletVelocity)
        
        for inletFace in self.faces['inlet']:
            self.calcInletVelocity(volumeVelocity, inletFace)

    def calcInletVelocity(self, volumeVelocity, inletFace):
        if self.parabolicInlet:
            vvC = volumeVelocity / inletFace.int_u
            inletVelocity = np.outer(vvC*inletFace.u, inletFace.normal)
            inletFace.inletVelocity = inletVelocity[~inletFace.bdyFlags,:].ravel()
        else:
            inletVelocity = (volumeVelocity / inletFace.inletArea) * inletFace.normal
            inletFace.inletVelocity = (np.ones((len(inletFace.appNodes), 3))*inletVelocity).ravel()

    def calcInscribeDiameters(self):
        """ Calculate the inscribe sphere diameter of the tetrohedron elements. """
        nodes = self.nodes
        elements = self.elementNodeIds
        inscribeDiameters = self.inscribeDiameters = np.empty(self.nElements)

        for iElm, elm in enumerate(elements):
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

    def calcOutletNeighbors(self, c):
        nodes = self.nodes
        elmNIds = self.elementNodeIds
        elmCenters = np.mean(nodes[elmNIds], axis=1)
        for outletFace in self.faces['outlet']:
            outletFace.neighbors = np.empty(len(outletFace.appNodes), dtype=int) # store element id
            outletFace.neighborsNs = np.empty((len(outletFace.appNodes), 4), dtype=float)
            for i,iOutlet in enumerate(outletFace.appNodes):
                # The aiming point iOutlet - c*n
                aimP = nodes[iOutlet] - c*outletFace.normal
                # Find where it belong, incidentally, calculate shape function (barycentric coordinates)
                # Find potential list
                distanceList = np.argsort(np.linalg.norm(elmCenters - aimP, axis=1))
                for iElm in distanceList:
                    lNIds = elmNIds[iElm]
                    # Calculate this elements inverse jacobian
                    jCol0 = nodes[lNIds[1]] - nodes[lNIds[0]]
                    jCol1 = nodes[lNIds[2]] - nodes[lNIds[0]]
                    jCol2 = nodes[lNIds[3]] - nodes[lNIds[0]]
                    # Cofactors
                    # +0,0  -0,1  +0,2
                    # -1,0  +1,1  -1,2
                    # +2,0  -2,1  +2,2
                    lCofac = np.array([[jCol1[1]*jCol2[2]-jCol1[2]*jCol2[1],
                                        jCol0[2]*jCol2[1]-jCol0[1]*jCol2[2],
                                        jCol0[1]*jCol1[2]-jCol0[2]*jCol1[1]],
                                       [jCol2[0]*jCol1[2]-jCol1[0]*jCol2[2],
                                        jCol0[0]*jCol2[2]-jCol2[0]*jCol0[2],
                                        jCol0[2]*jCol1[0]-jCol0[0]*jCol1[2]],
                                       [jCol1[0]*jCol2[1]-jCol1[1]*jCol2[0],
                                        jCol0[1]*jCol2[0]-jCol0[0]*jCol2[1],
                                        jCol0[0]*jCol1[1]-jCol1[0]*jCol0[1]]])
                    lDet = jCol0[0]*lCofac[0,0]+jCol1[0]*lCofac[0,1]+jCol2[0]*lCofac[0,2]
                    invJ = lCofac.T / lDet
                    # Local barycentric coordinates invJ*(x - x0)
                    lBcCoord = np.dot(invJ, aimP-nodes[lNIds[0]])

                    if np.all(lBcCoord >= 0.0) and np.all(lBcCoord <= 1.0):
                        
                        outletFace.neighbors[i] = iElm
                        outletFace.neighborsNs[i,0] = 1.0 - np.sum(lBcCoord)
                        outletFace.neighborsNs[i,1:] = lBcCoord

                        print('Find neighbor for {}, {}'.format(iOutlet, iElm))

                        break


    # def calcOutletNeighbors(self):

    #     # Rules to decide if is neighbor
    #     h = np.mean(self.inscribeDiameters)
    #     glbD = 3.0 * h
    #     rD = 2.4 * h
    #     zD = 2.4 * h

    #     nNeighbors = 4

    #     nodes = self.nodes
    #     labelIndices = np.arange(self.nNodes, dtype=int)
    #     labelIndicesTag = np.ones(self.nNodes, dtype=bool)
    #     labelIndicesTag[self.outlet] = False
    #     noOutletIndices = labelIndices[labelIndicesTag]

    #     outletNeighbors = [[] for _ in self.outlet]
    #     for i,iOutlet in enumerate(self.outlet):
    #         neighbors = self.findOutletNeighbors(i, iOutlet, nodes[noOutletIndices], nNeighbors, glbD, rD, zD)
    #         # if len(neighbors) < nNeighbors:
    #         #     print('First Failed to find neighbors for {}, neighbors {}'.format(iOutlet, neighbors))
    #         #     neighbors = self.findOutletNeighbors(i, iOutlet, nodes[noOutletIndices], 2*nNeighbors, 2.0*glbD, 2.0*rD, 2.0*zD)
    #         #     if len(neighbors) < nNeighbors:
    #         #         print('Failed to find neighbors for {}, neighbors {}'.format(iOutlet, neighbors))
    #         #         sys.exit(-1)

    #         # outletNeighbors[i].extend(np.array(neighbors, dtype=int))
    #         outletNeighbors[i].extend(noOutletIndices[neighbors])

    #     self.outletNeighbors = outletNeighbors

    # def findOutletNeighbors(self, i, iOutlet, wNodes, nNeighbors, glbD, rD, zD):
    #     hs = self.inscribeDiameters

    #     ol = self.nodes[iOutlet]
    #     # Get the neighbors in big range - potential neighbors
    #     # # pNghb = np.argwhere(np.linalg.norm(nodes - ol, axis=1) < glbD*hs[iOutlet])[:,0]
    #     # pNghb = np.argsort(np.linalg.norm(wNodes - ol, axis=1))[:3*nNeighbors]
    #     # rNghb = np.linalg.norm(wNodes[pNghb,:2] - ol[:2], axis=1) < rD
    #     # zNbDist = np.abs(wNodes[pNghb,-1] - ol[-1])
    #     # zNghb = np.logical_and(zNbDist > 0.3*hs[iOutlet], zNbDist < zD)

    #     # return pNghb[np.logical_and(rNghb, zNghb)]

    #     pNghb = np.argsort(np.linalg.norm(wNodes - ol, axis=1))[:3]
    #     return pNghb


    def DebugSaveNeighbors(self):

        # Save outlet too
        mark = np.zeros(self.nNodes)
        mark[self.outlet] = 13.0
        uTuples = numpy_to_vtk(mark)
        uTuples.SetName('outlet')
        self.polyDataModel.GetPointData().AddArray(uTuples)

        randomCheck = np.random.choice(len(self.outlet), 5)
        for iRdm in randomCheck:

            mark = np.zeros(self.nNodes)
            mark[self.outletNeighbors[iRdm]] = 3.0
            uTuples = numpy_to_vtk(mark)
            uTuples.SetName('{}_neighbor'.format(self.outlet[iRdm]))
            self.polyDataModel.GetPointData().AddArray(uTuples)
        
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName('Examples/CylinderProject/Results/neighbors.vtu')
        writer.Write()

    def DebugReadsInletVelocity(self):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName('Examples/CylinderProject/Results/inletVelocity.vtu')
        reader.Update()

        polyDataModel = reader.GetOutput()

        # Set the nodes and coordinates.
        self.dbgInletVelocity = vtk_to_numpy(polyDataModel.GetPointData().GetArray('velocity'))


    def Save(self, filename, counter, u, stress, uname='velocity'):
        """ Save the stress result of elements at time t with stress tensor of dim.
        """
        stressVec = numpy_to_vtk(stress)
        stressVec.SetName('pressure')
        self.polyDataModel.GetPointData().AddArray(stressVec)

        uTuples = numpy_to_vtk(u)
        uTuples.SetName(uname)
        self.polyDataModel.GetPointData().AddArray(uTuples)

        filename, fileExtension = os.path.splitext(filename)
        stressFilename = '{}{}{}'.format(filename, counter, fileExtension)

        writer = vtk.vtkXMLPolyDataWriter() if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(stressFilename)
        writer.Write()

        print('Write result to {}.'.format(stressFilename))

    # def DebugSave(self, filename, vals, uname=['debug'], pointData=[True]):
    #     for i, val in enumerate(vals):
    #         uTuples = numpy_to_vtk(val)
    #         uTuples.SetName(uname[i])
    #         if pointData[i]:
    #             self.polyDataModel.GetPointData().AddArray(uTuples)
    #         else:
    #             self.polyDataModel.GetCellData().AddArray(uTuples)

    #     writer = vtk.vtkXMLUnstructuredGridWriter() # if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
    #     writer.SetInputData(self.polyDataModel)
    #     writer.SetFileName(filename)
    #     writer.Write()

    #     print('Write result to {}.'.format(filename))

    def DebugSave(self, filename, counter, vals, uname=['debug'], pointData=[True]):
        for i, val in enumerate(vals):
            uTuples = numpy_to_vtk(val)
            uTuples.SetName(uname[i])
            if pointData[i]:
                self.polyDataModel.GetPointData().AddArray(uTuples)
            else:
                self.polyDataModel.GetCellData().AddArray(uTuples)

        filename, fileExtension = os.path.splitext(filename)
        stressFilename = '{}{}{}'.format(filename, counter, fileExtension)

        writer = vtk.vtkXMLUnstructuredGridWriter() # if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(stressFilename)
        writer.Write()

        print('Write result to {}.'.format(stressFilename))



class SolidMesh(Mesh):

    def __init__(self, comm, config, meshConfig, eqnConfig):
        Mesh.__init__(self, comm, config, meshConfig, eqnConfig)

        # ------------- Related to Calculation -----------------
        # Set the physical parameters.
        self.thickness = eqnConfig.thickness
        self.density = eqnConfig.density
        self.E = eqnConfig.E
        self.v = eqnConfig.v
        self.damp = eqnConfig.damp

        self.nSmp = meshConfig.nSmp
        self.vthickness = self.readProperties(self.thickness,
                                              eqnConfig.sigmaThickness,
                                              eqnConfig.rhoThickness,
                                              eqnConfig.thicknessFilename,
                                              meshConfig.file_path,
                                              meshConfig.regenerate_samples)
        self.vE = self.readProperties(self.E, eqnConfig.sigmaE, eqnConfig.rhoE,
                                      eqnConfig.YoungsModulusFilename,
                                      meshConfig.file_path,
                                      meshConfig.regenerate_samples)

        # Prepare the updated coordinates.
        self.originNodes = np.tile(self.nodes, (self.nSmp, 1, 1))
        self.updateNodes = self.originNodes

    def readProperties(self, mu, sigma, rho, propfilename, geofilename, regenerate=False):
        ''' Generate Gaussian Markov Random Fields for properties. '''
        # # Used for test, create with constant directly.
        # prop = np.full((self.nNodes, self.nSmp), mu)
        # filename, fileExtension = os.path.splitext(propfilename)
        # prop = np.load('{}{}{}'.format(filename, rho, fileExtension))


        if propfilename is None:
            prop = np.full((self.nNodes, self.nSmp), mu)

        else:
            filename, fileExtension = os.path.splitext(propfilename)
            filename = '{}{}{}'.format(filename, rho, fileExtension)

            # Read from file directly, if file does not exist, report error and exit!
            if os.path.exists(filename):
                prop = np.load(filename)
            else:
                print('Solid mesh, load random fields failed: File \'{}\' does not exist!'.format(filename))
                sys.exit()

            # # Use GMRF to generate the random fields.
            # if (not regenerate) and os.path.exists(propfilename): # if the file exists just read it from file.
            #     prop = np.load(filename)

            # else: # if not, create the random fields and write them to files.
            #     if self.rank == 0:
            #         prop = GMRF(geofilename, mu=mu, sigma=sigma, rho=rho, samplenum=self.nSmp, resfilename=filename)
            #     self.comm.Barrier()

            #     if self.rank != 0:
            #         prop = np.load(filename)
            #     self.comm.Barrier()

        return prop


    def processFaces(self):
        # Prepare the boundary indices for use directly.
        # self.boundary = np.append(self.faces['inlet'].glbNodeIds, self.faces['outlet'].glbNodeIds)
        # self.boundary = np.array([bdy.glbNodeIds for bdy in self.faces['boundaries']]).ravel()
        boundary = []
        for bdy in self.faces['boundaries']:
            boundary.extend(bdy.glbNodeIds)
        self.boundary = np.array(boundary)

    def setInitialConditions(self, iniCondConfig):
        # Set the velocity
        self.iniDu = self.setCondition(iniCondConfig.velocity, 'velocity')
        # Set the displacement
        self.iniU = self.setCondition(iniCondConfig.displacement, 'displacement')

    def setBoundaryCondtions(self, bdyCondConfig):
        # Set the traction
        # self.trac = self.setCondition(bdyCondConfig.traction, 'traction')
        self.bdyU = bdyCondConfig.bdyDisplacement

    def UpdateCoordinates(self, u):
        updateNodes = self.originNodes.reshape(self.nSmp, self.ndof) + u.transpose()
        self.updateNodes = updateNodes.reshape(self.nSmp, self.nNodes, self.dof)


    def Save(self, filename, counter, u, stress, uname='displacement'):
        """ Save the stress result of elements at time t with stress tensor of dim.
            Dim is an array like ['xx', 'yy', 'xy', 'xz', 'yz']
        """
        dim = np.array(['xx', 'yy', 'xy', 'xz', 'yz'])
        for i, name in enumerate(dim):
            for iSmp in range(self.nSmp):
                stressVec = numpy_to_vtk(stress[:,iSmp,i])
                stressVec.SetName('{}_{:03d}'.format(name, iSmp))
                self.polyDataModel.GetCellData().AddArray(stressVec)

        for iSmp in range(self.nSmp):
            uTuples = numpy_to_vtk(u[iSmp,:,:])
            uTuples.SetName('{}_{:03d}'.format(uname, iSmp))
            self.polyDataModel.GetPointData().AddArray(uTuples)

        filename, fileExtension = os.path.splitext(filename)
        stressFilename = '{}{}{}'.format(filename, counter, fileExtension)

        writer = vtk.vtkXMLPolyDataWriter() if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(stressFilename)
        writer.Write()

        print('Write result to {}.'.format(stressFilename))

    def SaveDisplacement(self, filename, counter, u):
        """ Save displacement and previous displacement to files.
            The previous displacement is used to update the coordinates.
        """
        for iSmp in range(self.nSmp):
            uTuples = numpy_to_vtk(u[iSmp,:,:])
            uTuples.SetName('{}_{:03d}'.format('u', iSmp))
            self.polyDataModel.GetPointData().AddArray(uTuples)

        filename, fileExtension = os.path.splitext(filename)
        stressFilename = '{}{}{}'.format(filename, counter, fileExtension)

        writer = vtk.vtkXMLPolyDataWriter() if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(stressFilename)
        writer.Write()

        print('Write result to {}.'.format(stressFilename))

    def DebugSave(self, filename, val, uname='debug', pointData=True):
        """ Save the stress result of elements at time t with stress tensor of dim.
            Dim is an array like ['xx', 'yy', 'xy', 'xz', 'yz']
        """

        uTuples = numpy_to_vtk(val)
        uTuples.SetName(uname)
        if pointData:
            self.polyDataModel.GetPointData().AddArray(uTuples)
        else:
            self.polyDataModel.GetCellData().AddArray(uTuples)

        writer = vtk.vtkXMLPolyDataWriter() # if fileExtension.endswith('vtp') else vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(filename)
        writer.Write()

        print('Write result to {}.'.format(filename))

