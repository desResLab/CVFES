#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Explicit VMS solver for fluid part.
"""

from mpi4py import MPI
import pyopencl as cl

from cvconfig import CVConfig
from mesh import *
from physicsSolver import PhysicsSolver

from timeit import default_timer as timer
import math


__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"

TAG_COMM_DOF = 211
TAG_COMM_DOF_VALUE = 212
# TAG_ELM_ID = 221
TAG_LHS = 224
TAG_STRESSES = 222
TAG_DISPLACEMENT = 223
TAG_NODE_ID = 224
# TAG_UNION = 224
TAG_CHECKING_STIFFNESS = 311


DOUBLE_NBYTES = 8


# Parameters for the explicit solver.
c1 = 4.0
c2 = 2.0
# c = 5.0*(11.7**2.0)
c = 5.0*11.7

vDof = 3
pDof = 1
dof = 4


class ExplicitVMSSolverGPUs(PhysicsSolver):
    """Explicit VMS method."""
    
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)

        # Initialize elements, nodes on local partition.
        self.lclNElms = mesh.nElements
        self.lclNNodes = mesh.lclNNodes
        self.lclElmNodeIds = mesh.lclElmNodeIds

        self.lclNCommNodes = mesh.lclNCommNodes
        self.lclNCommDof = self.lclNCommNodes * dof
        self.lclNSpecialHeadDof = mesh.lclNSpecialHead * dof
        self.lclBoundary = mesh.lclBoundary

        if self.InitializeGPU() < 0:
            exit(-1)

        self.InitializeParameters(config)
        self.InitializeSolver()

        # # Debugging.
        # # #1 Print out element node ids.
        # dbgElmID = 13
        # print('Element ID: {} \t Nodes\' IDs: {}'.format(
        #     self.mesh.elementsIds[dbgElmID], self.mesh.lclNodeIds[self.lclElmNodeIds[dbgElmID]]))
        # # #2 Save the volume infos into file.
        # # Collect the volumes from different color groups.
        # cpy_volumes_events = [None] * len(self.mesh.colorGroups)
        # lclVolumes = np.empty(self.lclNElms)
        # for iColorGrp in range(len(self.mesh.colorGroups)):
        # # for iColorGrp in range(1):
        #     grpVolumes = np.empty(len(self.mesh.colorGroups[iColorGrp]))
        #     cpy_volumes_events[iColorGrp] = cl.enqueue_copy(self.queue,
        #         grpVolumes, self.volumes_buf[iColorGrp])
        #     lclVolumes[self.mesh.colorGroups[iColorGrp]] = grpVolumes
        #     print(grpVolumes[0])
        # # Union the volumes across multi-GPUs.
        # print(lclVolumes)
        # glbVolumes = self.UnionCellValues(lclVolumes)
        # print(glbVolumes)
        # # Write to file.
        # if self.rank == 0:
        #     self.mesh.DebugSave('ExplicitVMSGPU_Debug.vtu', 0, [glbVolumes], ['volume'], [False])

        # exit(0)

        # # Debug for memory layout on GPU.
        # self.TestMemoryLayout()
        # exit(0)


    # def TestMemoryLayout(self):
    #     mem_flags = cl.mem_flags

    #     nRow = 4
    #     nColumns = 1000
    #     nBytes = nRow * nColumns * 8

    #     # Memory on GPU.
    #     gpuArray = cl.Buffer(self.context, mem_flags.READ_WRITE, nBytes)
    #     # Memory on CPU.
    #     cpuArray = np.empty((nRow, nColumns))
        
    #     test_event = self.program.test_memory_layout(self.queue, (nColumns,), (1,),
    #         np.int64(nRow), np.int64(nColumns), gpuArray)
    #     copy_event = cl.enqueue_copy(self.queue, cpuArray, gpuArray, wait_for=[test_event])
    #     copy_event.wait()

    #     print(cpuArray)


    def InitializeGPU(self):

        platforms = cl.get_platforms()

        devices = platforms[0].get_devices(cl.device_type.GPU)
        ndevices = len(devices)
        if ndevices < self.size:
            print('GPUs is not enough! Actural size: {}, need: {}'.format(ndevices, self.size))
            return -1

        self.device = devices[self.rank]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.localWorkSize = 64
        # assumes all the devices have same number of computes unit.
        self.num_compute_units = self.device.max_compute_units
        self.globalWorkSize = 8 * self.num_compute_units * self.localWorkSize
        print('gpu {} num of computing unites {}'.format(self.rank, self.num_compute_units))
        # calculating assignment
        self.num_groups = math.ceil(dof * self.lclNNodes / self.localWorkSize)

        # Read and build the kernel.
        kernelsource = open("explicitVMSSolverGPUs.cl").read()
        self.program = cl.Program(self.context, kernelsource).build()

        return 0


    def InitializeParameters(self, config):
        
        # Parameters for Tetrahedron
        alpha = 0.58541020
        beta = 0.13819660
        self.w = np.array([0.25, 0.25, 0.25, 0.25])
        self.lN = np.array([[alpha, beta, beta, beta],
                            [beta, alpha, beta, beta],
                            [beta, beta, alpha, beta],
                            [beta, beta, beta, alpha]])
        self.lDN = np.array([[-1.0, 1.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0, 1.0]])

        self.coefs = np.array([c1, c2, self.mesh.dviscosity, self.dt, c**2])

        # For ramp.
        self.constant_T = config.constant_T

    
    def InitializeSolver(self):

        # Memory size calculation.
        nElmBytes = 8 * self.lclNElms # double each element
        nNodeBytes = 8 * self.lclNNodes # double each node
        nCommNodeBytes = 8 * self.lclNCommNodes # double each comm node

        # Remember some variables in mesh.
        lclNodeIds = self.mesh.lclNodeIds
        
        # Copy nodes, elements info onto GPU.
        mem_flags = cl.mem_flags
        # nodes coordinates on local partition
        self.nodes_buf = cl.Buffer(self.context,
            mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
            hostbuf = self.mesh.nodes[self.mesh.lclNodeIds])
        # mesh coloring's color tags
        self.colorGps_buf = [cl.Buffer(self.context, 
            mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, 
            hostbuf = self.mesh.lclElmNodeIds[self.mesh.colorGroups[i]]) 
            for i in range(len(self.mesh.colorGroups))]
        # lN, lDN copy to GPU.
        self.lDN_buf = cl.Buffer(self.context,
            mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
            hostbuf = self.lDN)

        # # Another way.
        # nodes = np.copy(self.mesh.nodes[self.mesh.lclNodeIds].T)
        # print(nodes.shape)
        # self.nodes_buf = cl.Buffer(self.context,
        #     mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
        #     hostbuf = nodes)
        # self.colorGps_buf = [None] * len(self.mesh.colorGroups)
        # for iColorGrp in range(len(self.mesh.colorGroups)):
        #     elmNodeIds = np.copy(self.mesh.lclElmNodeIds[self.mesh.colorGroups[iColorGrp]].T)
        #     print(elmNodeIds.shape)
        #     self.colorGps_buf[iColorGrp] = cl.Buffer(self.context, 
        #     mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, 
        #     hostbuf = elmNodeIds)


        # Allocate memory on GPU for initial 'assemble'.
        self.volumes_buf = [cl.Buffer(self.context,
            mem_flags.READ_WRITE, 8*len(self.mesh.colorGroups[i]))
            for i in range(len(self.mesh.colorGroups))]
        self.DNs_buf = [cl.Buffer(self.context,
            mem_flags.READ_WRITE, 12*8*len(self.mesh.colorGroups[i]))
            for i in range(len(self.mesh.colorGroups))]
        
        self.lumpLHS_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, 4*nNodeBytes)

        # Allocate memory on CPU for synch lumped mass 'matrix'.
        lumpLHS = np.zeros((4, self.lclNCommNodes))

        # # Debugging for initial assemble.
        # dbgNodeCoords = np.empty((4, 3))
        # dbgNodeCoords_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, 12*8)
        # dbgDetJ = np.empty(1)
        # dbgDetJ_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, 8)
        
        # Initial assemble, calculate volume, DN for each element, assemble lumped LHS.
        initial_assemble_events = []
        for iColorGroup in range(len(self.colorGps_buf)):
        # for iColorGroup in range(1):
            nElms = len(self.mesh.colorGroups[iColorGroup])
            initial_assemble_event = self.program.initial_assemble(self.queue,
                (nElms,), (1,), np.int64(nElms), np.int64(self.lclNNodes), 
                self.nodes_buf, self.colorGps_buf[iColorGroup],
                self.volumes_buf[iColorGroup], self.DNs_buf[iColorGroup],
                self.lumpLHS_buf, wait_for=initial_assemble_events)
            initial_assemble_events = [initial_assemble_event]

            # cl.enqueue_copy(self.queue, dbgNodeCoords, dbgNodeCoords_buf,
            #     wait_for=initial_assemble_events)
            # cl.enqueue_copy(self.queue, dbgDetJ, dbgDetJ_buf,
            #     wait_for=initial_assemble_events)
            # print('Element ID: {} \n Nodes coords: {}\n determinant {}\n'.format(
            #     self.mesh.elementsIds[self.mesh.colorGroups[iColorGroup][0]],
            #     dbgNodeCoords, dbgDetJ))

            # varJac = dbgNodeCoords[1:,:] - dbgNodeCoords[0,:]
            # print('determinant is {}\n'.format(np.linalg.det(varJac.T)))


        initial_assemble_copy_events = [None] * dof
        for iCopy in range(dof):
            initial_assemble_copy_event = cl.enqueue_copy(self.queue, lumpLHS[iCopy], self.lumpLHS_buf,
                device_offset=iCopy*nNodeBytes, wait_for=initial_assemble_events)
            initial_assemble_copy_events[iCopy] = initial_assemble_copy_event

        cl.wait_for_events(initial_assemble_copy_events)

        # Synchronize the lumped mass 'matrix' for shared nodes btw partitions.
        self.SyncCommNodes(lumpLHS)
        # Copy the synchronized lumped mass 'matrix' back to GPU.
        initial_copy_events = [None] * dof
        for iCopy in range(dof):
            initial_copy_event = cl.enqueue_copy(self.queue, self.lumpLHS_buf, lumpLHS[iCopy],
                device_offset=iCopy*nNodeBytes)
            initial_copy_events[iCopy] = initial_copy_event
        cl.wait_for_events(initial_copy_events)


        # Allocate memory on GPU for (du, p) and (odu, op), and sdu.
        # (du, p) is mapping memory for the purpose of output result.
        map_flags = cl.map_flags
        self.pinned_res = cl.Buffer(self.context,
            mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, 4*nNodeBytes)
        self.res, _eventRes = cl.enqueue_map_buffer(self.queue, self.pinned_res,
            map_flags.WRITE | map_flags.READ, 0, (4, self.lclNNodes), lumpLHS.dtype)
        self.res[:3,:] = self.mesh.iniDu.reshape((self.mesh.nNodes, 3))[lclNodeIds,:].T # velocity
        self.res[-1,:] = self.mesh.iniP[lclNodeIds] # pressure

        self.res_buf = cl.Buffer(self.context,
            mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
            hostbuf = self.res)
        self.preRes_buf = cl.Buffer(self.context,
            mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
            hostbuf = self.res)
        self.sdu_buf = [cl.Buffer(self.context,
            mem_flags.READ_WRITE, 12*8*len(self.mesh.colorGroups[i]))
            for i in range(len(self.mesh.colorGroups))]

        # Allocate RHS buffer on GPU and the corresponding synchronize buffer on CPU.
        self.pinned_RHS = cl.Buffer(self.context,
            mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, 4*nCommNodeBytes)
        self.RHS, _eventRHS = cl.enqueue_map_buffer(self.queue, self.pinned_RHS,
            map_flags.WRITE | map_flags.READ, 0, (4, self.lclNCommNodes), lumpLHS.dtype)

        self.RHS_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, 4*nNodeBytes)

        # Allocate Dirichlet B.C. memory on CPU & GPU.
        if self.mesh.lclNInlet > 0:
            self.pinned_drchBCValue = cl.Buffer(self.context,
                mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR,
                3*8*self.mesh.lclNInlet)
            self.drchBCValue, _eventDrchBCValue = cl.enqueue_map_buffer(self.queue,
                self.pinned_drchBCValue, map_flags.WRITE | map_flags.READ, 0,
                (3, self.mesh.lclNInlet), lumpLHS.dtype)

        self.drchBCValue_buf = cl.Buffer(self.context,
            mem_flags.READ_WRITE, 3*8*len(self.lclBoundary))
        self.drchBCIndices_buf = cl.Buffer(self.context,
            mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
            hostbuf = self.lclBoundary)

        # Allocate external force buffer on GPU.
        # TODO:: Update when f is a real function of time and space !!!!!!!
        fs = self.mesh.f * np.ones((3, self.lclNNodes))
        self.fs_buf = cl.Buffer(self.context,
            mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
            hostbuf = fs)
        # Copy the parameters to GPU.
        self.params_buf = cl.Buffer(self.context,
            mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
            hostbuf = self.coefs)


    def Solve(self, t, dt):

        # Memory size calculation.
        nOffsetBytes = 8 * self.lclNNodes # double each comm node

        cl.enqueue_fill_buffer(self.queue, self.RHS_buf, np.float64(0.0), 0, self.res.nbytes)

        # Assemble the RHS for current time step.
        assemble_RHS_events = []
        for iColorGroup in range(len(self.colorGps_buf)):
            nElms = len(self.mesh.colorGroups[iColorGroup])
            assemble_RHS_event = self.program.assemble_RHS(self.queue,
                (nElms,), (1,), np.int64(nElms), np.int64(self.lclNNodes), 
                self.colorGps_buf[iColorGroup], self.fs_buf,
                self.volumes_buf[iColorGroup], self.DNs_buf[iColorGroup],
                self.res_buf, self.preRes_buf, self.sdu_buf[iColorGroup],
                self.params_buf, self.RHS_buf, wait_for=assemble_RHS_events)
            assemble_RHS_events = [assemble_RHS_event]

        assemble_copy_events = [None] * dof
        for iCopy in range(dof):
            assemble_copy_event = cl.enqueue_copy(self.queue, self.RHS[iCopy], self.RHS_buf,
                device_offset=iCopy*nOffsetBytes, wait_for=assemble_RHS_events)
            assemble_copy_events[iCopy] = assemble_copy_event

        cl.wait_for_events(assemble_copy_events)

        # Synchronize the RHS.
        self.SyncCommNodes(self.RHS)

        # Copy back the synchronized RHS common parts.
        sync_copy_events = [None] * dof
        for iCopy in range(dof):
            sync_copy_event = cl.enqueue_copy(self.queue, self.RHS_buf, self.RHS[iCopy],
                device_offset=iCopy*nOffsetBytes)
            sync_copy_events[iCopy] = sync_copy_event

        cl.wait_for_events(sync_copy_events)

        # Update the du, p for next time step, res=res+RHS/lumpLHS.
        self.preRes_buf, self.res_buf = self.res_buf, self.preRes_buf
        calc_res_event = self.program.calc_res(self.queue,
            (self.globalWorkSize,), (self.localWorkSize,), np.int64(self.num_groups),
            np.int64(dof*self.lclNNodes), np.float64(dt), self.RHS_buf, self.lumpLHS_buf,
            self.preRes_buf, self.res_buf)

        # Apply Dirichlet B.C.
        self.ApplyDirichletBCs(t+dt)


    def ApplyDirichletBCs(self, t):
        # Prepare bytes.
        nBdyNodeBytes = 8 * len(self.lclBoundary)
        # Remeber variables.
        lclInletValueIndices = self.mesh.lclInletValueIndices
        nInlet = len(self.mesh.inlet)

        update_drchBC_events = None

        if self.mesh.lclNInlet > 0:
            # Update the inlet velocity first.
            self.mesh.updateInletVelocity(t)
            # Combine the boundary condition at the start of each time step.
            inletDrchValues = np.empty(vDof*nInlet)
            nCount = 0
            if t > self.constant_T:
                for inlet in self.mesh.faces['inlet']:
                    inletDrchValues[nCount:nCount+len(inlet.appNodes)*vDof] = inlet.inletVelocity
                    nCount = nCount + len(inlet.appNodes)*vDof
            else:
                for inlet in self.mesh.faces['inlet']:
                    a = b = 0.5 * inlet.inletVelocity
                    n = math.pi/self.constant_T
                    inletDrchValues[nCount:nCount+len(inlet.appNodes)*vDof] = a - b*math.cos(n*t)
                    nCount = nCount + len(inlet.appNodes)*vDof

            # Assign to pinned_memory on CPU.
            self.drchBCValue[:,:] = inletDrchValues.reshape((nInlet, vDof))[lclInletValueIndices].T
            # Copy to GPU.
            update_drchBC_events = [None] * vDof
            for iCopy in range(vDof):
                update_drchBC_event = cl.enqueue_copy(self.queue,
                    self.drchBCValue_buf, self.drchBCValue[iCopy], device_offset=iCopy*nBdyNodeBytes)
                update_drchBC_events[iCopy] = update_drchBC_event

        apply_drchBC_event = self.program.apply_drchBC(self.queue,
            (len(self.lclBoundary),), (1,), np.int64(self.lclNNodes),
            self.drchBCValue_buf, self.drchBCIndices_buf, self.res_buf, wait_for=update_drchBC_events)

        apply_drchBC_event.wait()


    def Save(self, filename, counter):
        # Copy values back to CPU for saving.
        save_copy_event = cl.enqueue_copy(self.queue, self.res, self.res_buf)
        save_copy_event.wait()

        # Union the whole mesh's result.
        res = self.UnionNodes(self.res)

        if self.rank == 0:
            du = res[:vDof,:].T
            p = res[-1,:]
            
            self.mesh.Save(filename, counter, du, p)

    # def Save(self, filename, counter):
    #     cl.enqueue_copy(self.queue, self.res, self.res_buf)
    #     res = self.UnionNodes(self.res)
        
    #     # Debug save, lumpedLHS, RHS.
    #     lumpLHS = np.empty_like(self.res)
    #     cl.enqueue_copy(self.queue, lumpLHS, self.lumpLHS_buf)
    #     glbLumpLHS = self.UnionNodes(lumpLHS)

    #     RHS = np.empty_like(self.res)
    #     cl.enqueue_copy(self.queue, RHS, self.RHS_buf)
    #     glbRHS = self.UnionNodes(RHS)

    #     # Debug save to file.
    #     if self.rank == 0:
    #         lM = glbLumpLHS[0,:]
    #         vals = [lM, glbRHS.T, res[:vDof,:].T, res[-1,:]]
    #         uname = ['mass', 'rhs', 'velocity', 'pressure']
    #         pointData = [True, True, True, True]

    #         self.mesh.DebugSave(filename, counter, vals, uname, pointData)


    def SyncCommNodes(self, quant):
        """ Synchronize the quantity fo common nodes.
        """

        if self.size == 1:
            return

        totalCommNodeIds = self.mesh.totalCommNodeIds
        commNodeIds = self.mesh.commNodeIds
        commQuant = quant[:,:len(commNodeIds)]

        totalQuant = np.zeros((dof, len(totalCommNodeIds)))
        if self.rank == 0:

            # Add on self's (root processor's) quantity.
            indices = np.where(np.in1d(totalCommNodeIds, commNodeIds))[0]
            totalQuant[:,indices] += commQuant

            quantIdBuf = np.zeros(len(totalCommNodeIds), dtype=np.int64)
            quantBuf = np.empty(dof*len(totalCommNodeIds))
            recvInfo = MPI.Status()
            for i in range(1, self.size):
                self.comm.Recv(quantIdBuf, MPI.ANY_SOURCE, TAG_COMM_DOF, recvInfo)
                recvLen = recvInfo.Get_count(MPI.INT64_T)
                recvSource = recvInfo.Get_source()
                
                # quantBuf = np.empty(dof*recvLen)
                # Receive the quantity.
                self.comm.Recv(quantBuf, recvSource, TAG_COMM_DOF_VALUE, recvInfo)
                # TODO:: make sure the quant received length is consistent with quantIds'.

                # Add the quantity received to the totalQuant.
                indices = np.where(np.in1d(totalCommNodeIds, quantIdBuf[:recvLen]))[0]
                # totalQuant[:,indices] += quantBuf.reshape((dof, recvLen))
                totalQuant[:,indices] += quantBuf[:dof*recvLen].reshape((dof, recvLen))

        else:

            self.comm.Send(commNodeIds, 0, TAG_COMM_DOF)
            self.comm.Send(commQuant.ravel(), 0, TAG_COMM_DOF_VALUE)


        # Get the collected total quantities by broadcast.
        self.comm.Bcast(totalQuant, root=0)
        # Update the original quantity.
        indices = np.where(np.in1d(totalCommNodeIds, commNodeIds))[0]
        quant[:,:len(commNodeIds)] = totalQuant[:,indices]


    def UnionNodes(self, quant):

        nNodes = self.mesh.nNodes
        lclNNodes = self.lclNNodes
        lclNodeIds = self.mesh.lclNodeIds

        if self.size == 1:
            res = np.empty((dof, nNodes))
            res[:,lclNodeIds] = quant
            return res

        if self.rank == 0:

            res = np.empty((dof, nNodes))
            res[:,lclNodeIds] = quant

            nodesInfo = MPI.Status()
            idBuf = np.empty(nNodes, dtype=np.int64)
            resBuf = np.empty(dof*nNodes)
            
            for i in range(1, self.size):
                self.comm.Recv(idBuf, i, TAG_NODE_ID, nodesInfo) # MPI.ANY_SOURCE
                recvLen = nodesInfo.Get_count(MPI.INT64_T)
                nodesSource = nodesInfo.Get_source()
                ids = idBuf[:recvLen]

                # resBuf = np.empty((dof, len(ids)))
                self.comm.Recv(resBuf, nodesSource, TAG_DISPLACEMENT, nodesInfo)
                # res[:,ids] = resBuf
                res[:,ids] = resBuf[:dof*recvLen].reshape((dof, recvLen))

        else:
            self.comm.Send(lclNodeIds, 0, TAG_NODE_ID)
            self.comm.Send(quant, 0, TAG_DISPLACEMENT)
            res = None

        return res


    # def UnionCellValues(self, quant):
    #     """ For debugging """
    #     if self.size == 1:
    #         return quant

    #     gnElms = self.mesh.gnElements
    #     elmIds = self.mesh.elementsIds

    #     if self.rank == 0:
            
    #         res = np.zeros(gnElms)
    #         res[elmIds] = quant

    #         cellInfo = MPI.Status()
    #         idBuf = np.empty(gnElms, dtype=np.int64)
    #         resBuf = np.empty(gnElms)

    #         for i in range(1, self.size):
    #             self.comm.Recv(idBuf, i, TAG_NODE_ID, cellInfo)
    #             recvLen = cellInfo.Get_count(MPI.INT64_T)
    #             cellSource = cellInfo.Get_source()
    #             ids = idBuf[:recvLen]

    #             self.comm.Recv(resBuf, cellSource, TAG_DISPLACEMENT, cellInfo)
    #             res[ids] = resBuf[:recvLen]
    #     else:
    #         self.comm.Send(elmIds, 0, TAG_NODE_ID)
    #         self.comm.Send(quant, 0, TAG_DISPLACEMENT)
    #         res = None

    #     return res


class ExplicitVMSSolidSolverGPUs(PhysicsSolver):
    
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)







