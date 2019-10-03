import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sys
from timeit import default_timer as timer

LOOP_COUNT = 100

if __name__ == "__main__":

    # Get the arguments of the input model's name.
    mname = '../module_data/cylinder.npz'
    if len(sys.argv) > 1:
        mname = sys.argv[1]

    # Readin the matrix
    data = np.load(mname)
    indptr = data['indptr']
    indices = data['indices']
    M = data['M']
    NBytes = M.nbytes
    Shape = (len(indices), 3, 3)

    m = len(indptr)-1
    nSmp = 1000
    M = np.tile(M, (1, nSmp, 1, 1))
    v = np.ones((m*3, nSmp))
    y = np.zeros((m*3, nSmp))


    # # Self produced data to check accuracy.
    # indptr = np.array([0, 2, 5, 6, 7, 9, 10, 11, 12, 13])
    # indices = np.array([1, 2, 2, 5, 7, 0, 4, 3, 6, 2, 4, 2, 6])
    # Mc = np.arange(1.0, 10.0).reshape(3,3)
    # M = np.tile(Mc, (len(indices), 3, 1, 1))
    # # M = np.arange(351.0).reshape(len(indices), 3, 3, 3)
    # # NBytes = M.nbytes
    # Shape = (len(indices), 3, 3)

    # m = len(indptr)-1
    # nSmp = M.shape[1]
    # v = np.tile(np.array([1.0, 2.0, 3.0]), (nSmp, m)).T.copy()
    # # v = np.tile(np.arange(m*3), (nSmp, 1)).T.copy()
    # y = np.zeros((m*3, nSmp))


    # Setup the OpenCL environment.
    platform = cl.get_platforms()[0]

    device = platform.get_devices()[0]

    context = cl.Context([device])

    # Create 2 command-queues.
    queues = [cl.CommandQueue(context, device=device) for i in range(2)]

    mem_flags = cl.mem_flags
    indptr_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = indptr)
    indices_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = indices)
    # vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = v)

    # Allocate the OpenCL source and result buffer memory objects on GPU device GMEM.
    matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY, int(M.nbytes/nSmp))
    vector_buf = cl.Buffer(context, mem_flags.READ_ONLY, int(v.nbytes/nSmp))
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, int(y.nbytes/nSmp))

    # Allocate pinned source and result host buffers:
    #   Note: Pinned (Page Locked) memory is needed for async host<->GPU memory copy operations ***
    pinnedM = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, int(M.nbytes/nSmp))
    pinnedV = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, int(v.nbytes/nSmp))
    pinnedRes = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, int(y.nbytes/nSmp))

    # Get mapped pointers to pinned input host buffers.
    #   Note:  This allows general (non-OpenCL) host functions to access pinned buffers using standard pointers
    map_flags = cl.map_flags
    srcM, _eventSrcM = cl.enqueue_map_buffer(queues[0], pinnedM, map_flags.WRITE, 0, Shape, M.dtype)
    srcV, _eventSrcV = cl.enqueue_map_buffer(queues[0], pinnedV, map_flags.WRITE, 0, (m*3,), v.dtype)
    srcRes, _eventSrcRes = cl.enqueue_map_buffer(queues[0], pinnedRes, map_flags.READ, 0, (m*3,), y.dtype)


    halfSize = int(m/2)

    # Start with the most original one without any optimization.
    kernelsource = open("spMVSlicedOverlapping.cl").read()
    program = cl.Program(context, kernelsource).build()
    # mmul = program.mmul
    # mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])

    # localWorkSize = 256
    localWorkSize = 64
    num_compute_units = device.max_compute_units
    globalWorkSize = 8 * num_compute_units * localWorkSize
    print('num of computing unites {}'.format(num_compute_units))

    # Shared memory
    partialDotProduct = cl.LocalMemory(np.dtype(np.float64).itemsize * localWorkSize * 3)

    start = timer()

    for i in range(LOOP_COUNT):

        for iSmp in range(nSmp):

            # Fill the source with values.
            srcM[:,:,:] = M[:,iSmp,:,:]
            srcV[:] = v[:,iSmp]

            eventV = cl.enqueue_copy(queues[0], vector_buf, srcV)

            eventM0 = cl.enqueue_copy(queues[0], matrix_buf, srcM[:indptr[halfSize]], is_blocking=False)
            # spMVOverlapping
            matrix_dot_vector_kernel_event0 = \
            program.matrix_dot_vector(queues[0], (globalWorkSize,), (localWorkSize,), np.int64(halfSize), np.int64(0),
                                      indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf, partialDotProduct,
                                      wait_for=[eventM0])

            eventM1 = cl.enqueue_copy(queues[1], matrix_buf, srcM[indptr[halfSize]:], is_blocking=False,
                                      device_offset=indptr[halfSize]*9*8)
            # spMVOverlapping
            matrix_dot_vector_kernel_event1 = \
            program.matrix_dot_vector(queues[1], (globalWorkSize,), (localWorkSize,), np.int64(m), np.int64(halfSize),
                                      indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf, partialDotProduct,
                                      wait_for=[eventM1])


            ## Step #11. Move the kernel's output data to host memory.
            matrix_dot_vector_copy_event0 = \
            cl.enqueue_copy(queues[0], srcRes[:halfSize*3], destination_buf, is_blocking=False, wait_for=[matrix_dot_vector_kernel_event0])
            matrix_dot_vector_copy_event1 = \
            cl.enqueue_copy(queues[1], srcRes[halfSize*3:], destination_buf, is_blocking=False, wait_for=[matrix_dot_vector_kernel_event1], device_offset=halfSize*3*8)


            cl.wait_for_events([matrix_dot_vector_copy_event0, matrix_dot_vector_copy_event1])
            y[:,iSmp] = srcRes

    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))

    print(y)
