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


class GPUSolidSolver(PhysicsSolver):
    def __init__(self, comm, mesh, config):
        PhysicsSolver.__init__(self, comm, mesh, config)

        self.InitializeSync()
        if self.InitializeGPU() < 0:
            exit(-1)

        # Remember the fluid's time step,
        # used for read in stress from fluid solution
        # for segregated solvers.
        self.dt_f = config.dt_f
        self.stressFilename = config.exportBdyStressFilename # Global force
        self.useConstantStress = config.useConstantStress
        self.constant_T = config.constant_T
        self.constantPressure = config.constant_pressure # Local pressure
        if self.stressFilename is not None:
            if self.useConstantStress:
                self.etrac = np.load('{}.npy'.format(self.stressFilename))
            else:
                self.nt = 0
                self.strac = np.load('{}{}.npy'.format(self.stressFilename, self.nt))
                self.etrac = np.load('{}{}.npy'.format(self.stressFilename, self.nt+1))
        else:
            self.etrac = np.zeros((mesh.nNodes, 3))
            # self.etrac = np.zeros((mesh.ndof, 1))
        
        # self.appTrac = np.zeros((mesh.nNodes, 3))
        # # self.appTrac = np.zeros((mesh.ndof, 1))
        self.pressure = 0.0

        # Initialize the number of samples.
        self.nSmp = config.nSmp
        self.ndof = mesh.ndof
        self.nNodes = mesh.nNodes
        
        self.nElms = mesh.nElements
        self.lclNNodes = mesh.lclNNodes
        self.lclNodeIds = mesh.lclNodeIds
        self.lclElmNodeIds = mesh.lclElmNodeIds
        self.lclNDof = self.lclNNodes * mesh.dof # 3

        self.lclNCommNodes = mesh.lclNCommNodes
        self.lclNCommDof = self.lclNCommNodes * mesh.dof
        self.lclNSpecialHeadDof = mesh.lclNSpecialHead * mesh.dof
        self.lclBoundary = mesh.lclBoundary

        # Damp coef.
        self.damp = mesh.damp

        # Prepare the mesh info for union.
        self.dofs = np.array([[3*node, 3*node+1, 3*node+2] for node in self.lclNodeIds]).astype(int).ravel()

        # Initialize the context.
        self.du = mesh.iniDu[self.dofs] # velocity
        self.u = mesh.iniU[self.dofs] # displacement

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
        kernelsource = open("physicsSolverSDampGPUs.cl").read()
        self.program = cl.Program(self.context, kernelsource).build()

        return 0


    def InitializeSync(self):

        self.bdyDofs = np.array([[3*node, 3*node+1, 3*node+2] for node in self.mesh.lclBoundary]).astype(int).ravel()

        if self.size > 1:
            self.totalCommDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.totalCommNodeIds]).astype(int).ravel()
            self.commDofs = np.array([[i*3, i*3+1, i*3+2] for i in self.mesh.commNodeIds]).astype(int).ravel()


    def InitializeSolver(self):
        """ Calculate u_{-1} to start of the time looping.
            u_-1 = u_0 - dt*du_0 + 0.5*dt**2*ddu_0
        """
        # Allocate the np.array object in CPU.
        self.LM = np.zeros((self.lclNDof, self.nSmp)) # no synchronized
        self.LHS = np.zeros((self.lclNDof, self.nSmp)) # synchronized


        # Allocate the OpenCL source and result buffer memory objects on GPU device GMEM.
        mem_flags = cl.mem_flags

        self.nodes_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.nodes[self.lclNodeIds])
        # self.elmNodeIds_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.elementNodeIds)
        # mesh coloring's color tags
        self.colorGps_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.lclElmNodeIds[self.mesh.colorGroups[i]]) for i in range(len(self.mesh.colorGroups))]
        # self.colorGps_elmIds_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.colorGroups[i]) for i in range(len(self.mesh.colorGroups))]

        # for calculating M (mass) matrix, do not need to always exist in GPU memory
        thickness_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.mesh.vthickness[self.lclNodeIds])

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
        self.elmTE_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = elmTE[self.mesh.colorGroups[i]]) for i in range(len(self.mesh.colorGroups))]
        # self.elmE_buf = [cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = elmAveE[self.mesh.colorGroups[i]]) for i in range(len(self.mesh.colorGroups))]

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

        # cl.enqueue_fill_buffer(self.queue, self.LM_buf, np.float64(0.0), 0, self.LM.nbytes)
        # cl.enqueue_fill_buffer(self.queue, self.Ku_buf, np.float64(0.0), 0, self.LM.nbytes)
        # cl.enqueue_fill_buffer(self.queue, self.P_buf, np.float64(0.0), 0, self.LM.nbytes)

        map_flags = cl.map_flags
        self.appTrac_buf = cl.Buffer(self.context, mem_flags.READ_ONLY, int(self.lclNNodes*24))
        self.pinned_appTrac = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, int(self.lclNNodes*24))
        self.appTrac, _eventAppTrac = cl.enqueue_map_buffer(self.queue, self.pinned_appTrac, map_flags.WRITE, 0,
                                                            (self.lclNNodes, 3), self.LM.dtype)
        self.appTrac[:,:] = 0.0
        # prep_appTrac_event = cl.enqueue_copy(self.queue, self.appTrac_buf, self.appTrac)


        # 'Assemble' the inital M (mass) and Ku (stiffness) 'matrices'.
        # Kernel.
        initial_assemble_events = []
        for iColorGroup in range(len(self.colorGps_buf)):
            initial_assemble_event = \
            self.program.assemble_K_M_P(self.queue, (len(self.mesh.colorGroups[iColorGroup]),), (1,),
                                        np.int64(self.nSmp), np.float64(self.pressure),
                                        self.pVals_buf, self.nodes_buf, self.colorGps_buf[iColorGroup], thickness_buf,
                                        self.elmTE_buf[iColorGroup], u_buf, self.Ku_buf, self.LM_buf, self.P_buf,
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
        # Copy into GPU device and prepared.
        self.LHS_buf = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = self.LHS)


        # Calculate accelaration u''.
        # ddu = (F0 - C*du - Ku)/M
        self.ddu = np.zeros((self.lclNDof, self.nSmp))
        self.ddu_buf = cl.Buffer(self.context, mem_flags.READ_WRITE, self.LM.nbytes)
        initial_calc_ddu_event = \
        self.program.calc_ddu(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                              np.int64(self.nSmp), np.int64(self.lclNDof),
                              self.P_buf, self.Ku_buf, self.LHS_buf, self.ddu_buf)
        initial_ddu_copy_event = \
        cl.enqueue_copy(self.queue, self.ddu, self.ddu_buf, wait_for=[initial_calc_ddu_event])
        initial_ddu_copy_event.wait()
        # Synchronize the acceleration on common nodes.
        self.SyncCommNodes(self.ddu)
        # Add on the global force.
        self.ddu += self.appTrac.reshape(self.lclNDof, 1) / self.LHS


        # Prepare the memories.
        # Memory on GPU devices.
        map_flags = cl.map_flags
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
        prep_u_event = cl.enqueue_copy(self.queue, self.u_buf, self.srcU)


    def RefreshContext(self, physicSolver):

        t = physicSolver.t
        dt_f = self.dt_f
        ramp_T = self.constant_T

        # Set up the global force.
        if self.stressFilename is not None:
            if self.useConstantStress:
                if t > ramp_T:
                    appTrac = self.etrac
                else:
                    a = b = self.etrac/2.0
                    n = math.pi/self.constant_T
                    appTrac = a - b*math.cos(n*t)
            
            else:
                if t > ramp_T:
                    if int((t-ramp_T)/dt_f) > self.nt:
                        self.nt += 1
                        self.strac = self.etrac
                        self.etrac = np.load('{}{}.npy'.format(self.stressFilename, self.nt+1))
                        print('At t={} read in wallpressure_{}'.format(t, self.nt+1))

                    appTrac = self.strac + (t-ramp_T - self.nt*dt_f)*(self.etrac - self.strac)/dt_f
                
                else:
                    a = b = self.strac/2.0
                    n = math.pi/ramp_T
                    appTrac = a - b*math.cos(n*t)


            # Only use values of the dofs contained in the partition.
            self.appTrac[:,:] = appTrac[self.lclNodeIds,:]
            self.appTrac[self.lclBoundary,:] = 0.0

        # Set up the pressure.
        if t > self.constant_T:
            self.pressure = self.constantPressure
        else:
            a = b = self.constantPressure/2.0
            n = math.pi/self.constant_T
            self.pressure = a - b*math.cos(n*t)


    def Solve(self, t, dt):

        # start = timer()

        cl.enqueue_fill_buffer(self.queue, self.Ku_buf, np.float64(0.0), 0, self.LM.nbytes)
        cl.enqueue_fill_buffer(self.queue, self.P_buf, np.float64(0.0), 0, self.LM.nbytes)

        # end = timer()
        # print('--- Rank: {} time 0: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

        calc_Ku_events = []
        for iColorGrp in range(len(self.colorGps_buf)):
            calc_Ku_event = \
            self.program.assemble_K_P(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                                      np.int64(len(self.mesh.colorGroups[iColorGrp])),
                                      np.int64(self.nSmp), np.float64(self.pressure),
                                      self.pVals_buf, self.nodes_buf, self.colorGps_buf[iColorGrp],
                                      self.elmTE_buf[iColorGrp], self.u_buf, self.Ku_buf, self.P_buf,
                                      wait_for=calc_Ku_events)
            calc_Ku_events = [calc_Ku_event]

        # end = timer()
        # print('--- Rank: {} time 1: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

        calc_u_event = \
        self.program.calc_u(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                            np.int64(self.nSmp), np.int64(self.lclNDof),
                            np.float64(dt), np.float64(self.damp),
                            self.P_buf, self.Ku_buf, self.LM_buf, self.LHS_buf,
                            self.u_buf, self.up_buf, self.ures_buf, wait_for=[calc_Ku_event])
        # calc_u_event.wait() # TODO:: Comment off after debugging

        # end = timer()
        # print('--- Rank: {} time 2: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

        ures_copy_event = cl.enqueue_copy(self.queue, self.srcURes[:self.lclNCommDof], self.ures_buf,
                                          wait_for=[calc_u_event])
        # ures_copy_event.wait()

        # end = timer()
        # print('--- Rank: {} time 3: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

        # Synchronize the ures.
        self.SyncCommNodes(self.srcURes)

        # end = timer()
        # print('--- Rank: {} time 4: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

        # Apply boundary condition.
        self.ApplyBoundaryCondition(self.srcURes)
        # Enforce the applied boundary condition back to GPU. <lclNSpecialHeadDof>
        update_u_event = cl.enqueue_copy(self.queue, self.ures_buf, self.srcURes[:self.lclNSpecialHeadDof])

        # Add on the global force.
        appTrac_copy_event = cl.enqueue_copy(self.queue, self.appTrac_buf, self.appTrac)
        calc_u_event = \
        self.program.calc_u_appTrac(self.queue, (self.globalWorkSize,), (self.localWorkSize,),
                            np.int64(self.nSmp), np.int64(self.lclNDof), np.float64(dt),
                            self.LHS_buf, self.appTrac_buf, self.ures_buf,
                            wait_for=[update_u_event, appTrac_copy_event])
        calc_u_event.wait()


        # end = timer()
        # print('--- Rank: {} time 5: {:10.1f} ms'.format(self.rank, (end - start) * 1000.0))
        # start = timer()

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
        commQuant = quant[:len(commDofs)]

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
            self.comm.Send(commQuant.ravel(), 0, TAG_COMM_DOF_VALUE)


        # Get the collected total quantities by broadcast.
        self.comm.Bcast(totalQuant, root=0)
        # Update the original quantity.
        indices = np.where(np.isin(totalCommDofs, commDofs))[0]
        quant[:len(commDofs)] = totalQuant[indices]


    def Save(self, filename, counter):

        # Copy out the displacement from GPU to CPU.
        copy_u_event = cl.enqueue_copy(self.queue, self.srcU, self.u_buf)

        # Prepare/Union the displacement.
        resU = self.UnionDisplacement(self.srcU)

        if self.rank == 0:
            self.mesh.SaveDisplacement(filename, counter,
                                       resU.transpose().reshape(self.nSmp, self.mesh.nNodes, self.Dof))
        # Barrier everyone!
        self.comm.Barrier()


    def UnionDisplacement(self, quant):

        if self.size == 1:
            resU = np.empty((self.ndof, self.nSmp))
            resU[self.dofs,:] = quant
            return resU

        if self.rank == 0:

            resU = np.empty((self.ndof, self.nSmp))
            resU[self.dofs,:] = quant

            nodesInfo = MPI.Status()
            dofBuf = np.empty(self.ndof, dtype=np.int64)
            uBuf = np.zeros((self.ndof, self.nSmp))
            
            for i in range(1, self.size):
                self.comm.Recv(dofBuf, i, TAG_NODE_ID, nodesInfo) # MPI.ANY_SOURCE
                nodesSource = nodesInfo.Get_source()
                dofs = dofBuf[:nodesInfo.Get_count(MPI.INT64_T)]
                self.comm.Recv(uBuf, nodesSource, TAG_DISPLACEMENT, nodesInfo)
                # Flag the nodes uBuf acctually contains.
                resU[dofs,:] = uBuf[:len(dofs)]

        else:
            self.comm.Send(self.dofs, 0, TAG_NODE_ID)
            self.comm.Send(quant, 0, TAG_DISPLACEMENT)
            resU = None

        return resU

