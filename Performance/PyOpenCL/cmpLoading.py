import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from timeit import default_timer as timer
from util import *

LOOP_COUNT = 1000

if __name__ == "__main__":

    # Readin the matrix
    data = np.load('module_data/cylinder.npz')
    indptr = data['indptr']
    indices = data['indices']
    M = data['M']

    m = len(indptr)-1
    nSmp = 10
    M = np.tile(M, (1, nSmp, 1, 1))
    v = np.ones((nSmp, m*3))
    y = np.zeros((nSmp, m*3))

    # Transfer the M to the final structure.
    nM = np.zeros((nSmp, 9*M.shape[0]))
    Transfer(indptr, indices, M, nM)
    # print(nM)

    exIndptr = np.zeros(3*len(indptr)-2, dtype=int)
    exIndices = np.zeros(9*len(indices), dtype=int)
    ExtendStructureInfo(indptr, indices, exIndptr, exIndices)
    # print(exIndptr)
    # print(exIndices)

    LOCAL_SIZE = 1024
    rowBlocks = np.zeros(len(exIndptr)-1, dtype=int)
    nBlocks = CalcGPUAssignment(LOCAL_SIZE, exIndptr, rowBlocks)
    # print(nBlocks)
    # print(rowBlocks[:nBlocks])


    # Setup the OpenCL environment.
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])

    localWorkSize = 256
    # localWorkSize = 64
    num_compute_units = device.max_compute_units
    globalWorkSize = 16 * num_compute_units * localWorkSize


    # Start with the most original one without any optimization.
    kernelsource = open("loadingVector.cl").read()
    program = cl.Program(context, kernelsource).build()
    # mmul = program.mmul
    # mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])
    queue = cl.CommandQueue(context)

    # localWorkSize = 256
    # # localWorkSize = 64
    # num_compute_units = device.max_compute_units
    # globalWorkSize = 16 * num_compute_units * localWorkSize

    start = timer()

    for i in range(LOOP_COUNT):
        mem_flags = cl.mem_flags
        indptr_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = indptr)
        indices_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = indices)
        matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = M)
        vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = v)
        destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)

        matrix_dot_vector_kernel_event = \
        program.matrix_dot_vector(queue, (globalWorkSize,), (localWorkSize,), np.int64(m), np.int64(nSmp), indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf)

        ## Step #11. Move the kernel's output data to host memory.
        matrix_dot_vector_copy_event = \
        cl.enqueue_copy(queue, y, destination_buf, is_blocking=False, wait_for=[matrix_dot_vector_kernel_event])

        matrix_dot_vector_copy_event.wait()

    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))


    # Loading Streaming
    kernelsource = open("loadingStream.cl").read()
    program = cl.Program(context, kernelsource).build()
    # mmul = program.mmul
    # mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])
    queue = cl.CommandQueue(context)

    # localWorkSize = 256
    # # localWorkSize = 64
    # num_compute_units = device.max_compute_units
    # globalWorkSize = 16 * num_compute_units * localWorkSize

    start = timer()
    LDS = cl.LocalMemory(np.dtype(np.float64).itemsize * LOCAL_SIZE)

    for i in range(LOOP_COUNT):
        mem_flags = cl.mem_flags
        indptr_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = exIndptr)
        indices_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = exIndices)
        matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = nM)
        vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = v)
        destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.nbytes)

        row_blocks_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = rowBlocks[:nBlocks])
        matrix_dot_vector_kernel_event = \
        program.matrix_dot_vector(queue, (globalWorkSize,), (localWorkSize,), np.int64(nM.shape[1]), np.int64(nSmp), np.int64(v.shape[1]), np.int64(nBlocks-1), row_blocks_buf, indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf, LDS)

        ## Step #11. Move the kernel's output data to host memory.
        matrix_dot_vector_copy_event = \
        cl.enqueue_copy(queue, y, destination_buf, is_blocking=False, wait_for=[matrix_dot_vector_kernel_event])

        matrix_dot_vector_copy_event.wait()

    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))
