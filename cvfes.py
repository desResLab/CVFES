#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVFES class is the main solver of the CVFES project.
"""

from configobj import ConfigObj
from cvconfig import CVConfig
# from cvcomm import CVCOMM
from mesh import *
from solver import *
from mpi4py import MPI
from math import floor
from parmetis import PyParMETIS_V3_PartMeshKway

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


TAG_NODE_INFO   = 111
TAG_ELM_NUM     = 112
TAG_ELM         = 113
TAG_ELMID       = 114


class CVFES:

    def __init__(self, comm):

        self.comm = comm
        # For using convenient.
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

    def ReadInputFile(self, filename):
        # Read configuration file.
        self.cvConfig = CVConfig(ConfigObj(filename))
        # Loading meshes.
        self.meshes = [Mesh(msh) for msh in self.cvConfig.meshes]

    def Distribute(self, meshNo):
        """ Distribute the meshes between processors using ParMETIS lib. """

        # No need to distribute if only one processor being used.
        if self.size == 1:
            return 0

        # The mesh being distributed.
        mesh = self.meshes[meshNo]

        # Prepare the parameters used to call partMesh.
        # elmdist: contains the elements distributed btw procs initially.
        nAvg = floor(mesh.nElements / self.size)
        nHProcs = mesh.nElements - nAvg * self.size # Number of procs contains one more elm.
        elmdist = np.append((nAvg+1) * np.arange(nHProcs+1),
                            ((nAvg+1)*nHProcs) + nAvg * np.arange(1, self.size-nHProcs+1))
        elmdist = elmdist.astype(np.int64)
        # eptr: contains the head and end pointer to eind of each element.
        # eind: contains the initial element's node ids in each proc.
        ihead = elmdist[self.rank]
        itail = elmdist[self.rank+1]
        eptr = np.zeros(itail-ihead+1, dtype=np.int64)
        eind = np.empty(0, dtype=np.int64)
        for index, element in enumerate(mesh.elements[ihead:itail]):
            eptr[index+1] = eptr[index] + element.nNodes
            # For effeciency, this need to be changed to allocate space first, then assign in another loop
            eind = np.append(eind, element.nodes)

        # Prepare other parameters.
        tpwgts = np.ones(self.size) * (1.0/self.size)
        ubvec = np.array([1.05])
        options = np.array([0, 0, 0])

        # Call ParMETIS parttition mesh.
        (res, edgecut, part) = PyParMETIS_V3_PartMeshKway(
                                elmdist, eptr, eind,
                                ubvec=ubvec, tpwgts=tpwgts, options=options,
                                wgtflag=0, numflag=0,
                                ncon=1, nparts=self.size, ncommonnodes=2, # TODO:: Decide according to specific geometry!
                                comm=self.comm)

        if res != 1: # TODO:: Connect with METIS_OK constant.
            print('Calling ParMETIS_PartMeshKway failed!')
            return -1

        # DEBUG:
        # print('rank {} has part result {}\n'.format(self.rank, part))

        # Processor send the partition result to root,
        # then receive the elements belong to it from root.
        # Root proc receives all elements from each processor
        # and redistribute them according to the partitioning result.
        partids = np.arange(ihead, itail, dtype=np.int64)
        # Elements current processor owns.
        myElmsSize = int(mesh.nElements / self.size * 1.2)
        myElms = np.empty(myElmsSize, dtype=np.int64)

        if self.rank == 0:
            # Remember the whole partition result.
            mesh.partition = np.empty(mesh.nElements, dtype=np.int64)

            # Allocate the memory to store the unprocessed elms.
            recvElmsBuf = np.empty(mesh.nElements, dtype=np.int64)
            recvElmIdsBuf = np.empty(mesh.nElements, dtype=np.int64)
            recvElmsCounter = 0

            # Copy the root's partition result into mem first.
            partLength = len(part)
            recvElmsBuf[:partLength] = part
            recvElmIdsBuf[:partLength] = partids
            recvElmsCounter += partLength

            # Receive all 'other' elements from each processor.
            recvInfo = MPI.Status()
            for i in xrange(self.size-1):
                self.comm.Recv(recvElmsBuf[recvElmsCounter:], MPI.ANY_SOURCE, TAG_ELM, recvInfo)
                recvLen = recvInfo.Get_count(MPI.INT64_T)
                recvSource = recvInfo.Get_source()

                self.comm.Recv(recvElmIdsBuf[recvElmsCounter:], recvSource, TAG_ELMID, recvInfo)
                recvElmsCounter += recvLen

            print('root node collect {} elms, percentage {}.\n'.format(recvElmsCounter, float(recvElmsCounter)/mesh.nElements))

            # Root starts to process the collected data and split it to corresponding process.
            # For root node, pick up directly.
            elmsFlag = (recvElmsBuf == 0)
            elmsCounter = np.sum(elmsFlag)

            if elmsCounter > myElmsSize:
                addonSize = elmsCounter - myElmsSize
                myElms = np.append(myElms, np.empty(addonSize, dtype=np.int64))
                print('rank {} myElms has been extended.\n'.format(self.rank))

            myElms[:elmsCounter] = recvElmIdsBuf[elmsFlag] # This is what will be used finilly!
            # Remember the partition result.
            mesh.partition[recvElmIdsBuf[elmsFlag]] = 0

            for i in xrange(1, self.size):
                # Find the corresponding range of elms.
                pelmsFlag = (recvElmsBuf == i)

                # Start to send the elms to corresponding process.
                self.comm.Send(recvElmIdsBuf[pelmsFlag], dest=i, tag=TAG_ELMID)
                # Remeber the partition result.
                mesh.partition[recvElmIdsBuf[pelmsFlag]] = i
        else:
            # Other procs send the 'other' elements to root
            # and receive the ones belonging to itself from the root.
            self.comm.Send(part, dest=0, tag=TAG_ELM)
            self.comm.Send(partids, dest=0, tag=TAG_ELMID)

            # Receive the second part the elms that belong to the processor.
            recvInfo = MPI.Status()
            self.comm.Recv(myElms, 0, TAG_ELMID, recvInfo)
            elmsCounter = recvInfo.Get_count(MPI.INT64_T)

        self.comm.Barrier()

        myElms = myElms[:elmsCounter]

        # Update the mesh into sub-mesh in each processor,
        # notice that the [mesh] var acctually points to mesh in self.meshes.
        mesh.nElements = elmsCounter
        mesh.elements = mesh.elements[myElms]
        mesh.elementsMap = myElms

        # !!! After the distribution/partitioning no processor has all elements in the whole mesh again.

        # TODO:: Save the partition results into local files and read from files if existing.

        # Collect the common nodes between processors,
        # root collect nodes numbers from each processor and count, the node which counts more
        # than one will be set to the common and broadcast to all processors.
        # And each processor will recognize the common nodes it has according to the info it received.
        # Collect local nodes' numbers.
        myNodes = np.empty(3*mesh.nElements, dtype=np.int64)
        for iElm, elm in enumerate(mesh.elements):
            myNodes[3*iElm : 3*(iElm+1)] = elm.nodes
        myNodes = np.unique(myNodes)
        # Start to send and recv to filter the common nodes.
        if self.rank == 0:
            # Prepare the counter vector.
            nodesCounter = np.zeros(mesh.nNodes, dtype=int)
            # Start to count.
            nodesCounter[myNodes] += 1
            # Receive and count.
            nodesBuffer = np.empty(mesh.nNodes, dtype=np.int64)
            nodesInfo = MPI.Status()
            for i in xrange(1, self.size):
                self.comm.Recv(nodesBuffer, MPI.ANY_SOURCE, TAG_NODE_INFO, nodesInfo)
                nodesCounter[nodesBuffer[:nodesInfo.Get_count(MPI.INT64_T)]] += 1
            # Filter out the common nodes.
            commonNodes = np.where(nodesCounter > 1)[0]
            nCommon = len(commonNodes)
        else:
            self.comm.Send(myNodes, 0, TAG_NODE_INFO)
            nCommon = None

        # Broadcast the common nodes to everyone.
        nCommon = self.comm.bcast(nCommon, root=0)
        if self.rank != 0:
            commonNodes = np.empty(nCommon, dtype=np.int64)
        self.comm.Bcast(commonNodes, root=0)

        # Recognize the common nodes I contain.
        # mesh.commNodeIds = np.array(list(set(commonNodes).intersection(myNodes)))
        mesh.totalCommNodeIds = commonNodes
        mesh.commNodeIds = np.intersect1d(commonNodes, myNodes)


        # TODO:: Try to read the existing calculated results from local files, if exists start from there,
        #        if not start from initial conditions.


        # TODO:: Write the solution has been calculated into files when program has been cutoff accidentally.


    def Solve(self, meshNo):

        solverSwitcher = {
            'transient generalized-a': TransientGeneralizedASolver,
            'transient': TransientSolver
        }

        SolverClass = solverSwitcher.get(self.cvConfig.solver.method, None)

        if SolverClass is None:
            print('Unknown method: {}'.format(self.cvConfig.solver.method))
            return

        self.solver = SolverClass(self.comm, self.meshes[meshNo], self.cvConfig.solver)
        self.solver.Solve()

        # TODO:: Write back the calculation result.

    # ? Finalize
