#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Solver class provides the solver of the CVFES project.
    One Solver instance corresponds to one mesh and one method
    which can be decided by solver configuration.

    du: velocity
    p: pressure
    ddu: acceleration
    u: displacement
"""
from mpi4py import MPI
import pyopencl as cl

from cvconfig import CVConfig
from mesh import *
from shape import *
from physicsSolver import PhysicsSolver

from timeit import default_timer as timer

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"

TAG_COMM_DOF = 211
TAG_COMM_DOF_VALUE = 212
# TAG_ELM_ID = 221
TAG_LHS = 224
TAG_STRESSES = 222
TAG_DISPLACEMENT = 223
# TAG_UNION = 224
TAG_CHECKING_STIFFNESS = 311


DOUBLE_NBYTES = 8


class GPUSolidSolver(PhysicsSolver):
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)

        self.InitializeSync()
        if self.InitializeGPU() < 0:
            exit(-1)

        # Initialize the context.
        self.du = mesh.iniDu # velocity
        self.u = mesh.iniU # displacement
        self.appTraction = 0.0

        # Initialize the number of samples.
        self.nSmp = config.nSmp
        self.ndof = mesh.ndof
        self.nElms = mesh.nElements
        self.nNodes = mesh.nNodes

        # Calculate u_{-1} to start of the time looping.
        # u_-1 = u_0 - dt*du_0 + 0.5*dt**2*ddu_0
        self.InitializeSolver()

    def InitializeGPU(self):

        platforms = cl.get_platforms()

        devices = platforms[0].get_devices(cl.device_type.GPU)
        ndevices = len(devices)
        if ndevices < self.size:
            print('GPUs is not enough! Actural size: {}, need: {}'.format(ndevices, self.size))
            return -1

        self.device = devices[self.rank]
        self.context = cl.Context([self.device])
        # self.queues = [cl.CommandQueue(self.context) for i in range(2)]
        self.queue = cl.CommandQueue(self.context)

        self.localWorkSize = 64
        self.num_compute_units = self.device.max_compute_units # assumes all the devices have same number of computes unit.
        self.globalWorkSize = 8 * self.num_compute_units * self.localWorkSize
        print('gpu {} num of computing unites {}'.format(self.rank, self.num_compute_units))

        # Read and build the kernel.
        kernelsource = open("physicsSolverGPUs.cl").read()
        self.program = cl.Program(self.context, kernelsource).build()

        return 0

    def InitializeSync(self):

        self.bdyDofs = np.array([[3*node, 3*node+1, 3*node+2] for node in self.mesh.boundary]).astype(int).ravel()

        if self.size > 1:
            self.totalCommDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.totalCommNodeIds]).astype(int).ravel()
            self.commDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.commNodeIds]).astype(int).ravel()

    def InitializeSolver(self):
        """ Calculate u_{-1} to start of the time looping.
            u_-1 = u_0 - dt*du_0 + 0.5*dt**2*ddu_0
        """
        # Allocate the np.array object in CPU.
        self.LM = np.zeros((self.ndof, self.nSmp)) # no synchronized
        self.LHS = np.zeros((self.ndof, self.nSmp)) # synchronized


        # Allocate the OpenCL source and result buffer memory objects on GPU device GMEM.
        mem_flags = cl.mem_flags

        self.nodes_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.nodes)
        # self.elmNodeIds_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.elementNodeIds)
        # mesh coloring's color tags
        self.colorGps_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.elementNodeIds[self.mesh.colorGroups[i]]) for i in range(len(self.mesh.colorGroups))]
        self.colorGps_elmIds_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.colorGroups[i]) for i in range(len(self.mesh.colorGroups))]

        # for calculating M (mass) matrix, do not need to always exist in GPU memory
        thickness_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.vthickness)

        # for calculating K (stiffness) matrix, thicknessE (nElms, nSmp)
        # -- Young's Modulus
        elmVerE = self.mesh.vE[self.mesh.elementNodeIds,:]
        elmVerE = elmVerE.swapaxes(1,2)
        elmAveE = np.mean(elmVerE, axis=2)
        # -- thickness
        elmVerThick = self.mesh.vthickness[self.mesh.elementNodeIds,:]
        elmVerThick = elmVerThick.swapaxes(1,2)
        # elmAveThick = np.mean(elmVerThick, axis=2)
        # - thickness x E
        elmTE = np.mean(elmVerE*elmVerThick, axis=2)
        self.elmTE_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = elmTE)
        self.elmE_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = elmAveE)

        # for calculating K (stiffness) matrix, D needs
        k = 5.0/6.0
        v = self.mesh.v
        pVals = np.array([self.mesh.density, v, 0.5*(1.0-v), 0.5*k*(1.0-v), (1.0-v*v)])
        self.pVals_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = pVals)

        # The initial displacement b.c. (nNodes*3,)
        u_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.u)


        self.LM_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        self.Ku_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        self.P_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)


        # 'Assemble' the inital M (mass) and Ku (stiffness) 'matrices'.
        # Kernel.
        initial_assemble_events = []
        for iColorGroup in range(len(self.colorGps_buf)):
            initial_assemble_event = \
            self.program.assemble_K_M_P(self.queue, (len(self.mesh.colorGroups[iColorGroup]),), (1,),
                                        np.int64(self.nNodes), np.int64(self.nSmp), np.float64(self.appTraction),
                                        self.pVals_buf, self.nodes_buf, self.colorGps_buf[iColorGroup], thickness_buf,
                                        self.elmTE_buf, u_buf, self.Ku_buf, self.LM_buf, self.P_buf,
                                        wait_for=initial_assemble_events)
            initial_assemble_events = [initial_assemble_event]

        initial_assemble_copy_event = \
        cl.enqueue_copy(self.queue, self.LM, self.LM_buf, wait_for=initial_assemble_events)

        initial_assemble_copy_event.wait()


        # Synchronize the left-hand-side of each equition which is LM.
        # Copy the LM first to LHS.
        self.LHS[:,:] = self.LM
        # Synchronize.
        self.SyncCommNodes(self.LHS)
        self.UnionLHS()
        # Copy into GPU device and prepared.
        self.LHS_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.LHS)


        # Calculate accelaration u''.
        # ddu = (F0 - C*du - Ku)/M
        self.ddu = np.zeros((self.ndof, self.nSmp))
        self.ddu_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        initial_calc_ddu_event = \
        self.program.calc_ddu(self.queue, (self.globalWorkSize,), (self.localWorkSize,), np.int64(self.nSmp), np.int64(self.ndof),
                              self.P_buf, self.Ku_buf, self.LHS_buf, self.ddu_buf)
        initial_ddu_copy_event = \
        cl.enqueue_copy(self.queue, self.ddu, self.ddu_buf, wait_for=[initial_calc_ddu_event])
        initial_ddu_copy_event.wait()
        # Synchronize the acceleration on common nodes.
        self.SyncCommNodes(self.ddu)


        # Prepare the memories.
        # Memory on GPU devices.
        self.ures_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        self.u_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        self.up_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        self.stress_buf = cl.Buffer(self.context, mem_flags.WRITE_ONLY, int(self.nElms*self.nSmp*40))
        # Pinned memory on CPU.
        self.pinned_ures = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, self.LM.nbytes)
        self.pinned_u = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, self.LM.nbytes)
        self.pinned_up = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, self.LM.nbytes)
        self.pinned_stress = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, int(self.nElms*self.nSmp*40))
        # Map to CPU.
        map_flags = cl.map_flags
        self.srcURes, _eventSrcURes = cl.enqueue_map_buffer(self.queue, self.pinned_ures, map_flags.WRITE | map_flags.READ, 0,
                                                            self.LM.shape, self.LM.dtype)
        self.srcU, _eventSrcU = cl.enqueue_map_buffer(self.queue, self.pinned_u, map_flags.WRITE | map_flags.READ, 0,
                                                            self.LM.shape, self.LM.dtype)
        self.srcUP, _eventSrcUP = cl.enqueue_map_buffer(self.queue, self.pinned_up, map_flags.WRITE | map_flags.READ, 0,
                                                            self.LM.shape, self.LM.dtype)
        self.stress, _eventStress = cl.enqueue_map_buffer(self.queue, self.pinned_stress, map_flags.READ, 0,
                                                            (self.nElms, self.nSmp, 5), self.LM.dtype)


        # Use Taylor Expansion to get u_-1.
        self.srcU[:,:] = self.u[np.newaxis].transpose()
        self.srcUP[:,:] = self.srcU - self.dt * self.du[np.newaxis].transpose() + self.dt**2 * self.ddu / 2.0
        # copy up first to device
        prep_up_event = cl.enqueue_copy(self.queue, self.up_buf, self.srcUP)


    def ApplyPressure(self, appTraction):
        self.appTraction = appTraction


    def Solve(self, t, dt):

        cl.enqueue_fill_buffer(self.queue, self.Ku_buf, np.float64(0.0), 0, self.LM.nbytes)
        cl.enqueue_fill_buffer(self.queue, self.P_buf, np.float64(0.0), 0, self.LM.nbytes)

        update_u_event = cl.enqueue_copy(self.queue, self.u_buf, self.srcU)

        calc_Ku_events = [update_u_event]
        for iColorGrp in range(len(self.colorGps_buf)):
            calc_Ku_event = \
            self.program.assemble_K_P(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                                      np.int64(len(self.mesh.colorGroups[iColorGrp])),
                                      np.int64(self.nNodes), np.int64(self.nSmp), np.float64(self.appTraction),
                                      self.pVals_buf, self.nodes_buf, self.colorGps_buf[iColorGrp],
                                      self.elmTE_buf, self.u_buf, self.Ku_buf, self.P_buf,
                                      wait_for=calc_Ku_events)
            calc_Ku_events = [calc_Ku_event]

        calc_u_event = \
        self.program.calc_u(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                            np.int64(self.nSmp), np.int64(self.ndof), np.float64(dt),
                            self.P_buf, self.Ku_buf, self.LM_buf, self.LHS_buf,
                            self.u_buf, self.up_buf, self.ures_buf, wait_for=[calc_Ku_event])

        ures_copy_event = cl.enqueue_copy(self.queue, self.srcURes, self.ures_buf, wait_for=[calc_u_event])
        ures_copy_event.wait()

        # Synchronize the ures.
        self.SyncCommNodes(self.srcURes)
        self.ApplyBoundaryCondition(self.srcURes)

        # Update/Shift the pointers.
        self.srcURes, self.srcU, self.srcUP = self.srcUP, self.srcURes, self.srcU
        self.ures_buf, self.u_buf, self.up_buf = self.up_buf, self.ures_buf, self.u_buf

    def ApplyBoundaryCondition(self, quant): # TODO:: Change to according to configuration.
        quant[self.bdyDofs,:] = self.mesh.bdyU


    def SyncCommNodes(self, quant):
        """ Synchronize the quantity fo common nodes.
        """

        if self.size == 1:
            return

        totalCommDofs = self.totalCommDofs
        commDofs = self.commDofs
        commQuant = quant[commDofs]

        totalQuant = np.zeros((len(totalCommDofs), self.nSmp))
        if self.rank == 0:

            # Add on self's (root processor's) quantity.
            indices = np.where(np.isin(totalCommDofs, commDofs))[0]
            totalQuant[indices] += commQuant

            quantIdBuf = np.zeros(len(totalCommDofs), dtype=np.int64)
            quantBuf = np.zeros(len(totalCommDofs)*self.nSmp)
            recvInfo = MPI.Status()
            for i in range(1, self.size):
                self.comm.Recv(quantIdBuf, MPI.ANY_SOURCE, TAG_COMM_DOF, recvInfo)
                recvLen = recvInfo.Get_count(MPI.INT64_T)
                recvSource = recvInfo.Get_source()
                # Receive the quantity.
                self.comm.Recv(quantBuf, recvSource, TAG_COMM_DOF_VALUE, recvInfo)
                # TODO:: make sure the quant received length is consistent with quantIds'.

                # Add the quantity received to the totalQuant.
                indices = np.where(np.isin(totalCommDofs, quantIdBuf[:recvLen]))[0]
                totalQuant[indices] += quantBuf[:recvLen*self.nSmp].reshape(recvLen, self.nSmp)

        else:

            self.comm.Send(commDofs, 0, TAG_COMM_DOF)
            self.comm.Send(commQuant.flatten(), 0, TAG_COMM_DOF_VALUE)


        # Get the collected total quantities by broadcast.
        self.comm.Bcast(totalQuant, root=0)
        # Update the original quantity.
        indices = np.where(np.isin(totalCommDofs, commDofs))[0]
        quant[commDofs] = totalQuant[indices]

    def UnionLHS(self):

        if self.size == 1:
            return

        if self.rank == 0:

            lhsBuf = np.zeros((self.ndof, self.nSmp))
            for i in range(1, self.size):
                self.comm.Recv(lhsBuf, MPI.ANY_SOURCE, TAG_LHS)
                # Flag the nodes uBuf acctually contains.
                self.LHS[lhsBuf!=0] = lhsBuf[lhsBuf!=0]

        else:
            self.comm.Send(self.LHS, 0, TAG_LHS)

        self.comm.Bcast(self.LHS, root=0)


    def Save(self, filename, counter):
        # Prepare/Union the displacement.
        self.UnionDisplacement()

        # Prepare stress.
        update_u_event = cl.enqueue_copy(self.queue, self.u_buf, self.srcU)

        calc_stress_events = []
        for iColorGrp in range(len(self.colorGps_buf)):
            calc_s_event = \
            self.program.calc_stress(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                                      np.int64(len(self.mesh.colorGroups[iColorGrp])),
                                      np.int64(self.nNodes), np.int64(self.nSmp),
                                      self.pVals_buf, self.nodes_buf,
                                      self.colorGps_buf[iColorGrp], self.colorGps_elmIds_buf[iColorGrp],
                                      self.elmE_buf, self.up_buf, self.u_buf, self.stress_buf,
                                      wait_for=[update_u_event])
            calc_stress_events.append(calc_s_event)

        stress_copy_event = cl.enqueue_copy(self.queue, self.stress, self.stress_buf, wait_for=calc_stress_events)
        stress_copy_event.wait()

        self.UnionStress()

        if self.rank == 0:
            self.mesh.Save(filename, counter,
                           self.srcU.transpose().reshape(self.nSmp, self.mesh.nNodes, self.Dof),
                           self.glbStresses)
        # Barrier everyone!
        self.comm.Barrier()

    def UnionDisplacement(self):

        if self.size == 1:
            return

        if self.rank == 0:

            uBuf = np.zeros((self.ndof, self.nSmp))
            for i in range(1, self.size):
                self.comm.Recv(uBuf, MPI.ANY_SOURCE, TAG_DISPLACEMENT)
                # Flag the nodes uBuf acctually contains.
                self.srcU[uBuf!=0] = uBuf[uBuf!=0]

        else:
            self.comm.Send(self.srcU, 0, TAG_DISPLACEMENT)


    def UnionStress(self):

        if self.size == 1:
            self.glbStresses = self.stress
            return

        if self.rank == 0:

            self.glbStresses = np.zeros((self.mesh.gnElements, self.nSmp, 5))
            self.glbStresses[self.mesh.partition==0, :, :] = self.stress

            bufSize = int(self.mesh.gnElements / self.size * 1.2)
            stressesBuf = np.empty((bufSize, self.nSmp, 5))

            recvInfo = MPI.Status()
            for i in range(1, self.size):
                # Receive the stresses from each processor.
                self.comm.Recv(stressesBuf, i, TAG_STRESSES, recvInfo) # MPI.ANY_SOURCE
                recvLen = recvInfo.Get_count(MPI.DOUBLE)
                # p = recvInfo.Get_source()
                # Assign.
                self.glbStresses[self.mesh.partition==i, :, :] = stressesBuf[:int(recvLen/self.nSmp/5), :, :]
        else:

            self.comm.Send(self.stress, 0, TAG_STRESSES)
