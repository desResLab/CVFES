import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from timeit import default_timer as timer
import queue
import sys
from mpi4py import MPI

LOOP_COUNT = 1000


def main(mname):

    # Readin the matrix
    data = np.load(mname)
    indptr = data['indptr']
    indices = data['indices']
    M = data['M']

    m = len(indptr)-1
    nSmp = 1000
    M = np.tile(M, (1, nSmp, 1, 1))
    v = np.ones((m*3, nSmp))
    y = np.zeros((m*3, nSmp))

    # Get the ranks.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    # Slice the data to be processed.
    srow = rank * int(m / size)
    erow = (rank + 1) * int(m / size)
    if rank == size - 1:
        erow = m
    lm = erow - srow

    tM = M[indptr[srow]:indptr[erow]]
    ty = y[srow*3:erow*3]
    tIndptr = indptr[srow:erow+1] - indptr[srow]
    tIndices = indices[indptr[srow]:indptr[erow]]


    platforms = cl.get_platforms()

    devices = platforms[0].get_devices(cl.device_type.GPU)
    ndevices = len(devices)
    if ndevices < size:
        print('GPUs is not enough! Actural size: {}, need: {}'.format(ndevices, size))
        return

    device = devices[rank]
    context = cl.Context([device])
    queues = [cl.CommandQueue(context) for i in range(2)]


    # Create the buffers.
    mem_flags = cl.mem_flags
    indptr_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tIndptr)
    indices_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tIndices)

    # Allocate the OpenCL source and result buffer memory objects on GPU device GMEM.
    matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY, tM.nbytes)
    vector_buf = cl.Buffer(context, mem_flags.READ_ONLY, v.nbytes)
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, ty.nbytes)

    # Allocate pinned source and result host buffers:
    #   Note: Pinned (Page Locked) memory is needed for async host<->GPU memory copy operations ***
    pinnedM = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, tM.nbytes)
    pinnedV = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, v.nbytes)
    pinnedRes = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, ty.nbytes)

    # Get mapped pointers to pinned input host buffers.
    #   Note:  This allows general (non-OpenCL) host functions to access pinned buffers using standard pointers
    map_flags = cl.map_flags
    srcM, _eventSrcM = cl.enqueue_map_buffer(queues[0], pinnedM, map_flags.WRITE, 0, tM.shape, tM.dtype)
    srcV, _eventSrcV = cl.enqueue_map_buffer(queues[0], pinnedV, map_flags.WRITE, 0, v.shape, v.dtype)
    srcRes, _eventSrcRes = cl.enqueue_map_buffer(queues[0], pinnedRes, map_flags.READ, 0, ty.shape, ty.dtype)

    srcM[:,:,:,:] = tM
    srcV[:,:] = v

    halfSize = int(lm / 2)

    localWorkSize = 64
    num_compute_units = device.max_compute_units # assumes all the devices have same number of computes unit.
    globalWorkSize = 8 * num_compute_units * localWorkSize
    print('gpu {} num of computing unites {}'.format(rank, num_compute_units))


    # Read and build the kernel.
    kernelsource = open("multiGPUsTest.cl").read()
    program = cl.Program(context, kernelsource).build()

    start = timer()

    for iloop in range(LOOP_COUNT):

        eventV = cl.enqueue_copy(queues[0], vector_buf, srcV)

        eventM0 = cl.enqueue_copy(queues[0], matrix_buf, srcM[:tIndptr[halfSize]])
        # Kernel.
        matrix_dot_vector_kernel_event0 = \
        program.matrix_dot_vector(queues[0], (globalWorkSize,), (localWorkSize,),
                                  np.int64(halfSize), np.int64(nSmp), np.int64(0),
                                  indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf,
                                  wait_for=[eventV, eventM0])


        eventM1 = cl.enqueue_copy(queues[1], matrix_buf, srcM[tIndptr[halfSize]:],
                                  device_offset=tIndptr[halfSize]*nSmp*9*8)
        # Kernel.
        matrix_dot_vector_kernel_event1 = \
        program.matrix_dot_vector(queues[1], (globalWorkSize,), (localWorkSize,),
                                  np.int64(lm), np.int64(nSmp), np.int64(halfSize),
                                  indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf,
                                  wait_for=[eventV, eventM1])

        ## Step #11. Move the kernel's output data to host memory.
        matrix_dot_vector_copy_event0 = \
        cl.enqueue_copy(queues[0], srcRes[:halfSize*3], destination_buf,
                        is_blocking=False,
                        wait_for=[matrix_dot_vector_kernel_event0])

        matrix_dot_vector_copy_event1 = \
        cl.enqueue_copy(queues[1], srcRes[halfSize*3:], destination_buf,
                        is_blocking=False,
                        wait_for=[matrix_dot_vector_kernel_event1],
                        device_offset=halfSize*3*nSmp*8)

        # matrix_dot_vector_copy_event0.wait()
        cl.wait_for_events([matrix_dot_vector_copy_event0, matrix_dot_vector_copy_event1])

    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))

    # myRes = np.load('y.npy')
    # print(np.array_equal(myRes, srcRes))
    # print(myRes - srcRes)
    # print(srcRes)


if __name__ == "__main__":

    # Read in data.
    mname = '../module_data/cylinder.npz'
    if len(sys.argv) > 1:
        mname = sys.argv[1]

    main(mname)
