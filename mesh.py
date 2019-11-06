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
            self.glbNodeIds = np.where(np.in1d(tglbNodeIds, glbNodeIds))[0]
            # Add in elements info, only used by 'inlet' face now.
            self.nElements = polyDataModel.GetNumberOfCells()
            self.elements = np.array([ElementOfFace(polyDataModel.GetCell(i)) for i in range(self.nElements)])


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
        # Collect all the inlet and outlet
        self.wall = np.array([w.glbNodeIds for w in self.faces['wall']]).ravel()

        for inletFace in self.faces['inlet']:
            inlet = inletFace.glbNodeIds
            flags = np.isin(inlet, self.wall)
            intersectNodes = inlet[flags]
            inletFace.appNodes = inlet[~flags]

            # Calculate the inlet area used for calculating BC velocity.
            inletArea = 0.0
            for iElm, elm in enumerate(inletFace.elements):
                numIncluding = np.sum(~np.isin(inlet[elm.nodes], intersectNodes))
                inletArea += elm.area * numIncluding / 3.0
            inletFace.inletArea = inletArea

        # Set the inlet to be appliable inlet glbNodeIds
        self.inlet = np.array([il.appNodes for il in self.faces['inlet']]).ravel()

    def setInitialConditions(self, iniCondConfig):
        # Set the acceleration
        self.iniDDu = self.setCondition(iniCondConfig.acceleration, 'acceleration')
        # Set the velocity
        self.iniDu = self.setCondition(iniCondConfig.velocity, 'velocity')
        # Set the pressure
        self.iniP = self.setCondition(iniCondConfig.pressure, 'pressure', dof=1)

    def setBoundaryCondtions(self, bdyCondConfig):
        # Set the inlet velocity BC.
        for inletFace in self.faces['inlet']:
            nInlet = len(inletFace.appNodes)
            velocity = bdyCondConfig.inletVelocity / inletFace.inletArea
            inletFace.inletVelocity = self.setCondition(velocity, 'velocity', n=nInlet, valueDof=2) # z-axis

        # Set the outlet Natural BC.
        # TODO:: Find out how to Apply Natrual BC !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for outletFace in self.faces['outlet']:
            nOutlet = len(outletFace.glbNodeIds)
            outletFace.ouletH = self.setCondition(bdyCondConfig.outletH, 'h', n=nOutlet, dof=1)

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

        # if propfilename is None:
        #     prop = np.full((self.nNodes, self.nSmp), mu)

        # else:
        #     filename, fileExtension = os.path.splitext(propfilename)
        #     filename = '{}{}_{}{}'.format(filename, self.nSmp, rho, fileExtension)

        #     if (not regenerate) and os.path.exists(propfilename): # if the file exists just read it from file.
        #         prop = np.load(filename)

        #     else: # if not, create the random fields and write them to files.
        #         if self.rank == 0:
        #             prop = GMRF(geofilename, mu=mu, sigma=sigma, rho=rho, samplenum=self.nSmp, resfilename=filename)
        #         self.comm.Barrier()

        #         if self.rank != 0:
        #             prop = np.load(filename)
        #         self.comm.Barrier()

        prop = np.full((self.nNodes, self.nSmp), mu)

        # filename, fileExtension = os.path.splitext(propfilename)
        # prop = np.load('{}{}{}'.format(filename, rho, fileExtension))

        return prop


    def processFaces(self):
        # Prepare the boundary indices for use directly.
        # self.boundary = np.append(self.faces['inlet'].glbNodeIds, self.faces['outlet'].glbNodeIds)
        self.boundary = np.array([bdy.glbNodeIds for bdy in self.faces['boundaries']]).ravel()

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

