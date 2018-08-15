#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVFES class is the main solver of the CVFES project.
"""
import numpy as np

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

        self.area = vtkCell.ComputeArea()

        # TODO::Add edges info if needed later.

class Mesh:
    """ Mesh structure, contains nodes and elements primarily.
        Right now one vtp file corresponds to one mesh and one domain.
    """

    def __init__(self, filename, domainId):

        reader = vtk.vtkXMLPolyDataReader() if filename.endswith('vtp') else vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        polyDataModel = reader.GetOutput()

        # Set the nodes and coordinates.
        self.nNodes = polyDataModel.GetNumberOfPoints() # _nNodes
        vtkNodes = polyDataModel.GetPoints().GetData()
        self.nodes = vtk_to_numpy(vtkNodes) # _nodes

        # Set the element groups.
        # Will be updated to sub-group after partition.
        self.nElements = polyDataModel.GetNumberOfCells()
        self.elements = np.array([Element(polyDataModel.GetCell(i)) for i in xrange(self.nElements)])
        self.elememtsMap = None

        # Set total number of elements in the mesh.
        self.gnElements = self.nElements

        # Set the domain id.
        self.domainId = domainId


