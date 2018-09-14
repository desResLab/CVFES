#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVFES class is the main solver of the CVFES project.
"""
from cvconfig import CVConfig
import numpy as np
from os.path import splitext

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class Element:

    def __init__(self, vtkCell):

        self.nNodes = vtkCell.GetNumberOfPoints()
        self.nodes = np.zeros(self.nNodes, dtype=np.int64)
        for ipt in xrange(self.nNodes):
            self.nodes[ipt] = vtkCell.GetPointId(ipt)

        # self.area = vtkCell.ComputeArea() # Need to change, because element can update the displacement!

        # TODO::Add edges info if needed later.


class Mesh:
    """ Mesh structure, contains nodes and elements primarily.
        Right now one vtp file corresponds to one mesh and one domain.
    """

    def __init__(self, meshConfig):

        reader = vtk.vtkXMLPolyDataReader() if meshConfig.file_path.endswith('vtp') else vtk.vtkUnstructuredGridReader()
        reader.SetFileName(meshConfig.file_path)
        reader.Update()

        self.polyDataModel = reader.GetOutput()

        # Set the nodes and coordinates.
        self.nNodes = self.polyDataModel.GetNumberOfPoints() # _nNodes
        vtkNodes = self.polyDataModel.GetPoints().GetData()
        self.nodes = vtk_to_numpy(vtkNodes) # _nodes
        self.commNodeIds = None # common nodes processor contains
        self.totalCommNodeIds = None

        # Set the element groups.
        # Will be updated to sub-group after partition.
        self.nElements = self.polyDataModel.GetNumberOfCells()
        self.elements = np.array([Element(self.polyDataModel.GetCell(i)) for i in xrange(self.nElements)])
        self.elememtsMap = None

        # Set total number of elements in the mesh.
        self.gnElements = self.nElements
        self.partition = None # The whole partition result, only used by root.

        # Set the domain id.
        self.domainId = meshConfig.domainId
        # Set the physical parameters.
        self.density = meshConfig.density
        self.E = meshConfig.E
        self.v = meshConfig.v

        # Set the total number of degree of freedoms.
        self.ndof = 3 * self.nNodes

        # Set the boundary.
        self.setBoundary()

        # Set the initial conditions.
        # TODO:: might need to reconsider how to organize the structure of configuration,
        #        maybe open the initial condition file only once is enough.
        self.setInitialConditions(meshConfig.initialConditions)
        self.setBoundaryCondtions(meshConfig.boundaryConditions)

        # Set the result filename.
        self.stressFilename = meshConfig.stressFilename

    def setInitialConditions(self, iniCondConfig):
        # Set the acceleration
        self.iniDDu = self.setCondition(iniCondConfig.acceleration, 'acceleration')
        # Set the velocity
        self.iniDu = self.setCondition(iniCondConfig.velocity, 'velocity')
        # Set the pressure
        self.iniP = self.setCondition(iniCondConfig.pressure, 'pressure')
        # Set the displacement
        self.iniU = self.setCondition(iniCondConfig.displacement, 'displacement')

    def setBoundaryCondtions(self, bdyCondConfig):
        # Set the traction
        # TODO:: Switch btw getting traction from file and uniform value.
        # self.trac = self.setCondition(bdyCondConfig.traction, 'traction')
        self.traction = bdyCondConfig.traction

    def setCondition(self, value, fieldname=None):
        """ Setting the initial conditions of the mesh based on the configuration,
            if it's presetting uniform value then set it to the whole geometry
            otherwise read it from geometry (vtp) files.
        """
        if isinstance(value, str):

            reader = vtk.vtkXMLPolyDataReader() if value.endswith('vtp') else vtk.vtkUnstructuredGridReader()
            reader.SetFileName(value)
            reader.Update()

            polyDataModel = reader.GetOutput()

            # TODO:: check the compatibility between initial condition file and the mesh's geometry

            prop = vtk_to_numpy(polyDataModel.GetPointData().GetArray(fieldname))

        else:
            prop = np.zeros(self.ndof)
            prop[:] = value

        return prop

    def setBoundary(self): # TODO:: The way to decide where is boundary.
        tol = 0.01

        Z = self.nodes[:, 2]
        self.boundary = np.where((abs(Z-np.amax(Z))<tol) | (abs(Z-np.amin(Z))<tol))[0]

    def UpdateCoordinates(self, u):
        self.nodes += np.reshape(u, (self.nNodes, 3))

    def Save(self, t, dim, stress, u=None):
        """ Save the stress result of elements at time t with stress tensor of dim.
            Dim is an array like ['xx', 'yy', 'xy', 'xz', 'yz']
        """
        for i, name in enumerate(dim):
            stressVec = numpy_to_vtk(stress[:,i])
            stressVec.SetName(name)
            self.polyDataModel.GetCellData().AddArray(stressVec)

        if u is not None:
            uTuples = numpy_to_vtk(u)
            uTuples.SetName('displacement')
            self.polyDataModel.GetPointData().AddArray(uTuples)

        filename, fileExtension = splitext(self.stressFilename)
        stressFilename = '{}{}{}'.format(filename, t, fileExtension)

        writer = vtk.vtkXMLPolyDataWriter() if fileExtension.endswith('vtp') else vtk.vtkUnstructuredGridWriter()
        writer.SetInputData(self.polyDataModel)
        writer.SetFileName(stressFilename)
        writer.Write()


