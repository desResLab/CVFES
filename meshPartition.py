import numpy as np
from math import floor
import os.path
from mpi4py import MPI
from parmetis import PyParMETIS_V3_PartMeshKway

from mesh import *

TAG_NODE_INFO   = 111
TAG_ELM_NUM     = 112
TAG_ELM         = 113
TAG_ELMID       = 114


def CalcLocalInfo(size, mesh):
    """ Calculate local node Ids from the partitioning result. """

    # Local node ids contained in each partation.
    lclNodeIds = np.sort(np.unique(mesh.elementNodeIds.ravel()))

    mesh.lclNNodes = len(lclNodeIds)
    mesh.lclNodeIds = np.empty(mesh.lclNNodes, dtype=int)
    # 1. First chunk is the common nodes' ids.
    mesh.lclNCommNodes = len(mesh.commNodeIds)
    mesh.lclNodeIds[:mesh.lclNCommNodes] = mesh.commNodeIds
    # 2. Second chunk is the boundary nodes' ids if have any.
    lclBdyFlag = np.in1d(lclNodeIds, mesh.boundary)
    mesh.lclNBoundary = np.sum(lclBdyFlag)
    # Length of the beginning chunk that needs to transfer back and forth btw GPUs and CPUs.
    mesh.lclNSpecialHead = mesh.lclNCommNodes + mesh.lclNBoundary
    mesh.lclNodeIds[mesh.lclNCommNodes:mesh.lclNSpecialHead] = lclNodeIds[lclBdyFlag]
    # 3. Fill up the rest with the normal nodes that can always stay in GPUs.
    lclNormalFlag = np.logical_and(~np.in1d(lclNodeIds, mesh.commNodeIds), ~lclBdyFlag)
    mesh.lclNodeIds[mesh.lclNSpecialHead:] = lclNodeIds[lclNormalFlag]

    # Elemental local node ids.
    sorter = np.argsort(mesh.lclNodeIds)
    mesh.lclElmNodeIds = sorter[np.searchsorted(mesh.lclNodeIds, mesh.elementNodeIds, sorter=sorter)]

    # Mesh boundary local ids.
    mesh.lclBoundary = np.where(np.in1d(mesh.lclNodeIds, mesh.boundary))[0]


def MeshPartition(name, comm, mesh):
    """ Distribute the meshes between processors using ParMETIS lib. """

    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if size == 1:
        # Assign global info to local directly.
        mesh.lclNCommNodes = 0
        
        mesh.lclNNodes = mesh.nNodes
        mesh.lclNodeIds = np.empty(mesh.lclNNodes, dtype=int)
        mesh.lclNSpecialHead = mesh.lclNBoundary = len(mesh.boundary)
        mesh.lclNodeIds[:mesh.lclNBoundary] = mesh.boundary
        lclNodeIds = np.arange(mesh.lclNNodes, dtype=int)
        mesh.lclNodeIds[mesh.lclNBoundary:] = lclNodeIds[~np.in1d(lclNodeIds, mesh.boundary)]

        sorter = np.argsort(mesh.lclNodeIds)
        mesh.lclElmNodeIds = sorter[np.searchsorted(mesh.lclNodeIds, mesh.elementNodeIds, sorter=sorter)]

        mesh.lclBoundary = np.where(np.in1d(mesh.lclNodeIds, mesh.boundary))[0]
        return 0


    if os.path.exists('{}.npz'.format(name)):
        data = np.load('{}.npz'.format(name), allow_pickle=True)

        # assignment
        mesh.nElements = data['nElms']
        mesh.elements = data['elms']
        mesh.elementsIds = data['elmIds']
        mesh.elementNodeIds = data['elmNodeIds']
        mesh.totalCommNodeIds = data['totalCommNodeIds']
        mesh.commNodeIds = data['commNodeIds']
        mesh.partition = data['partition']

        CalcLocalInfo(size, mesh)

        return 0

    nElms = mesh.nElements
    elms = mesh.elements

    # Prepare the parameters used to call partMesh.
    # elmdist: contains the elements distributed btw procs initially.
    nAvg = floor(nElms / size)
    nHProcs = nElms - nAvg * size # Number of procs contains one more elm.
    elmdist = np.append((nAvg+1) * np.arange(nHProcs+1),
                        ((nAvg+1)*nHProcs) + nAvg * np.arange(1, size-nHProcs+1))
    elmdist = elmdist.astype(np.int64)
    # eptr: contains the head and end pointer to eind of each element.
    # eind: contains the initial element's node ids in each proc.
    ihead = elmdist[rank]
    itail = elmdist[rank+1]
    eptr = np.zeros(itail-ihead+1, dtype=np.int64)
    eind = np.empty(0, dtype=np.int64)
    for index, element in enumerate(elms[ihead:itail]):
        eptr[index+1] = eptr[index] + element.nNodes
        # For effeciency, this need to be changed to allocate space first, then assign in another loop
        eind = np.append(eind, element.nodes)

    # Prepare other parameters.
    tpwgts = np.ones(size) * (1.0/size)
    ubvec = np.array([1.05])
    options = np.array([0, 0, 0])

    # print('rank {} elmdist {} tpwgts {}'.format(self.rank, elmdist, tpwgts))

    # Call ParMETIS parttition mesh.
    (res, edgecut, part) = PyParMETIS_V3_PartMeshKway(
                            elmdist, eptr, eind,
                            ubvec=ubvec, tpwgts=tpwgts, options=options,
                            wgtflag=0, numflag=0,
                            ncon=1, nparts=size, ncommonnodes=2, # TODO:: Decide according to specific geometry!
                            comm=comm)

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
    myElmsSize = int(nElms / size * 1.2)
    myElms = np.empty(myElmsSize, dtype=np.int64)

    if rank == 0:
        # Remember the whole partition result.
        mesh.partition = np.empty(nElms, dtype=np.int64)

        # Allocate the memory to store the unprocessed elms.
        recvElmsBuf = np.empty(nElms, dtype=np.int64)
        recvElmIdsBuf = np.empty(nElms, dtype=np.int64)
        recvElmsCounter = 0

        # Copy the root's partition result into mem first.
        partLength = len(part)
        recvElmsBuf[:partLength] = part
        recvElmIdsBuf[:partLength] = partids
        recvElmsCounter += partLength

        # Receive all 'other' elements from each processor.
        recvInfo = MPI.Status()
        for i in range(1, size):
            comm.Recv(recvElmsBuf[recvElmsCounter:], i, TAG_ELM, recvInfo) # MPI.ANY_SOURCE
            recvLen = recvInfo.Get_count(MPI.INT64_T)
            # recvSource = recvInfo.Get_source()

            comm.Recv(recvElmIdsBuf[recvElmsCounter:], i, TAG_ELMID, recvInfo) # recvSource
            recvElmsCounter += recvLen

        # print('root node collect {} elms, percentage {}.\n'.format(recvElmsCounter, float(recvElmsCounter)/mesh.nElements))

        # Root starts to process the collected data and split it to corresponding process.
        # For root node, pick up directly.
        elmsFlag = (recvElmsBuf == 0)
        elmsCounter = np.sum(elmsFlag)

        if elmsCounter > myElmsSize:
            addonSize = elmsCounter - myElmsSize
            myElms = np.append(myElms, np.empty(addonSize, dtype=np.int64))
            print('rank {} myElms has been extended.\n'.format(rank))

        myElms[:elmsCounter] = recvElmIdsBuf[elmsFlag] # This is what will be used finilly!
        # Remember the partition result.
        mesh.partition[recvElmIdsBuf[elmsFlag]] = 0

        for i in range(1, size):
            # Find the corresponding range of elms.
            pelmsFlag = (recvElmsBuf == i)

            # Start to send the elms to corresponding process.
            comm.Send(recvElmIdsBuf[pelmsFlag], dest=i, tag=TAG_ELMID)
            # Remeber the partition result.
            mesh.partition[recvElmIdsBuf[pelmsFlag]] = i
    else:
        # Other procs send the 'other' elements to root
        # and receive the ones belonging to itself from the root.
        comm.Send(part, dest=0, tag=TAG_ELM)
        comm.Send(partids, dest=0, tag=TAG_ELMID)

        # Receive the second part the elms that belong to the processor.
        recvInfo = MPI.Status()
        comm.Recv(myElms, 0, TAG_ELMID, recvInfo)
        elmsCounter = recvInfo.Get_count(MPI.INT64_T)

    comm.Barrier()

    myElms = myElms[:elmsCounter]

    # Update the mesh into sub-mesh in each processor,
    # notice that the [mesh] var acctually points to mesh in self.meshes.
    mesh.nElements = elmsCounter
    mesh.elements = mesh.elements[myElms]
    mesh.elementsIds = myElms
    mesh.elementNodeIds = np.array([elm.nodes for elm in mesh.elements])

    # !!! After the distribution/partitioning no processor has all elements in the whole mesh again.


    # Collect the common nodes between processors,
    # root collect nodes numbers from each processor and count, the node which counts more
    # than one will be set to the common and broadcast to all processors.
    # And each processor will recognize the common nodes it has according to the info it received.
    # Collect local nodes' numbers.

    myNodes = np.sort(np.unique(mesh.elementNodeIds.ravel()))
    # Start to send and recv to filter the common nodes.
    if rank == 0:
        # Prepare the counter vector.
        nodesCounter = np.zeros(mesh.nNodes, dtype=int)
        # Start to count.
        nodesCounter[myNodes] += 1
        # Receive and count.
        nodesBuffer = np.empty(mesh.nNodes, dtype=np.int64)
        nodesInfo = MPI.Status()
        for i in range(1, size):
            comm.Recv(nodesBuffer, MPI.ANY_SOURCE, TAG_NODE_INFO, nodesInfo)
            nodesCounter[nodesBuffer[:nodesInfo.Get_count(MPI.INT64_T)]] += 1
        # Filter out the common nodes.
        commonNodes = np.where(nodesCounter > 1)[0]
        nCommon = len(commonNodes)
    else:
        comm.Send(myNodes, 0, TAG_NODE_INFO)
        nCommon = None

    # Broadcast the common nodes to everyone.
    nCommon = comm.bcast(nCommon, root=0)
    if rank != 0:
        commonNodes = np.empty(nCommon, dtype=np.int64)
    comm.Bcast(commonNodes, root=0)

    # Recognize the common nodes I contain.
    # mesh.commNodeIds = np.array(list(set(commonNodes).intersection(myNodes)))
    mesh.totalCommNodeIds = commonNodes
    mesh.commNodeIds = np.intersect1d(commonNodes, myNodes)

    CalcLocalInfo(size, mesh)

    # Save the partition results into local files and read from files if existing.
    np.savez(name, nElms=mesh.nElements, elms=mesh.elements,
             elmIds=mesh.elementsIds, elmNodeIds=mesh.elementNodeIds,
             totalCommNodeIds=mesh.totalCommNodeIds, commNodeIds=mesh.commNodeIds,
             partition=mesh.partition)

    return 0
