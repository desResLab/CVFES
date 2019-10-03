import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sys
from timeit import default_timer as timer

LOOP_COUNT = 1


# MAX_NBytes = 4*1073741824 # 4GB
MAX_NBytes = int(0.5*1073741824) # 0.5GB

def fitSamples(maxNSmp, nSmp):
    leftSmps = nSmp
    slicedSmp = [0]
    while leftSmps > maxNSmp:
        slicedSmp.append(slicedSmp[-1]+maxNSmp)
        leftSmps -= maxNSmp
    slicedSmp.append(nSmp)
    return slicedSmp


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

    m = len(indptr)-1
    nSmp = 100
    M = np.tile(M, (1, nSmp, 1, 1))
    v = np.ones((m*3, nSmp))
    y = np.zeros((m*3, nSmp))


    # # Self produced data to check accuracy.
    # indptr = np.array([0, 2, 5, 6, 7, 9, 10, 11, 12, 13])
    # indices = np.array([1, 2, 2, 5, 7, 0, 4, 3, 6, 2, 4, 2, 6])
    # Mc = np.arange(1.0, 10.0).reshape(3,3)
    # M = np.tile(Mc, (len(indices), 1000, 1, 1))
    # # M = np.arange(351.0).reshape(len(indices), 3, 3, 3)

    # m = len(indptr)-1
    # nSmp = M.shape[1]
    # v = np.tile(np.array([1.0, 2.0, 3.0]), (nSmp, m)).T.copy()
    # # v = np.tile(np.arange(m*3), (nSmp, 1)).T.copy()
    # y = np.zeros((m*3, nSmp))


    # Calculate how much to slice.
    MAX_NSmp = int(MAX_NBytes / (M.shape[0] * 72))
    MAX_NSmp_Bytes = MAX_NSmp * M.shape[0] * 72
    slicedSmp = fitSamples(MAX_NSmp, nSmp)
    nslicedSmp = len(slicedSmp) - 1
    print(slicedSmp)


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
    # indptr_buf = cl.Buffer(context, mem_flags.READ_ONLY, indptr.nbytes)
    # indices_buf = cl.Buffer(context, mem_flags.READ_ONLY, indices.nbytes)
    matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY, MAX_NSmp_Bytes)
    vector_buf = cl.Buffer(context, mem_flags.READ_ONLY, v.shape[0]*MAX_NSmp*8)
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, y.shape[0]*MAX_NSmp*8)

    # Allocate pinned source and result host buffers:
    #   Note: Pinned (Page Locked) memory is needed for async host<->GPU memory copy operations ***
    # pinnedIndptr = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, indptr.nbytes)
    # pinnedIndices = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, indices.nbytes)
    pinnedM = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, M.nbytes)
    pinnedV = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, v.nbytes)
    pinnedRes = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.ALLOC_HOST_PTR, y.nbytes)

    # Get mapped pointers to pinned input host buffers.
    #   Note:  This allows general (non-OpenCL) host functions to access pinned buffers using standard pointers
    map_flags = cl.map_flags
    # srcIndptr, _eventSrcIndptr = cl.enqueue_map_buffer(queues[0], pinnedIndptr, map_flags.WRITE, 0, indptr.shape, indptr.dtype)
    # srcIndices, _eventSrcIndices = cl.enqueue_map_buffer(queues[0], pinnedIndices, map_flags.WRITE, 0, indices.shape, indices.dtype)
    srcM, _eventSrcM = cl.enqueue_map_buffer(queues[0], pinnedM, map_flags.WRITE, 0, M.shape, M.dtype)
    srcV, _eventSrcV = cl.enqueue_map_buffer(queues[0], pinnedV, map_flags.WRITE, 0, v.shape, v.dtype)
    srcRes, _eventSrcRes = cl.enqueue_map_buffer(queues[0], pinnedRes, map_flags.READ, 0, y.shape, y.dtype)

    # Initialize
    srcM[:,:,:,:] = M
    srcV[:,:] = v

    # cl.enqueue_copy(queues[0], indptr_buf, srcIndptr)
    # cl.enqueue_copy(queues[0], indices_buf, srcIndices)

    halfSize = int(m/2)

    # Start with the most original one without any optimization.
    kernelsource = open("spMVSlicedOverlapping1.cl").read()
    program = cl.Program(context, kernelsource).build()
    # mmul = program.mmul
    # mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])

    # localWorkSize = 256
    localWorkSize = 64
    num_compute_units = device.max_compute_units
    globalWorkSize = 8 * num_compute_units * localWorkSize
    print('num of computing unites {}'.format(num_compute_units))


    start = timer()

    for i in range(LOOP_COUNT):

        for iSliced in range(nslicedSmp):

            nlocalSmp = slicedSmp[iSliced+1]-slicedSmp[iSliced]

            # # Fill the source with values.
            # srcM[:,:nlocalSmp,:,:] = M[:,slicedSmp[iSliced]:slicedSmp[iSliced+1],:,:]
            # srcV[:,:nlocalSmp] = v[:,slicedSmp[iSliced]:slicedSmp[iSliced+1]]


            eventV0 = cl.enqueue_copy(queues[0], vector_buf, srcV[:,slicedSmp[iSliced]:slicedSmp[iSliced+1]])

            eventM0 = cl.enqueue_copy(queues[0], matrix_buf,
                                      srcM[:indptr[halfSize],slicedSmp[iSliced]:slicedSmp[iSliced+1]],
                                      is_blocking=False)
            # spMVOverlapping
            matrix_dot_vector_kernel_event0 = \
            program.matrix_dot_vector(queues[0], (globalWorkSize,), (localWorkSize,),
                                      np.int64(halfSize), np.int64(0), np.int64(nlocalSmp), np.int64(nlocalSmp),
                                      indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf, wait_for=[eventM0])

            eventM1 = cl.enqueue_copy(queues[1], matrix_buf,
                                      srcM[indptr[halfSize]:,slicedSmp[iSliced]:slicedSmp[iSliced+1]],
                                      is_blocking=False,
                                      device_offset=indptr[halfSize]*nlocalSmp*9*8)
            # spMVOverlapping
            matrix_dot_vector_kernel_event1 = \
            program.matrix_dot_vector(queues[1], (globalWorkSize,), (localWorkSize,),
                                      np.int64(m), np.int64(halfSize), np.int64(nlocalSmp), np.int64(nlocalSmp),
                                      indptr_buf, indices_buf, matrix_buf, vector_buf, destination_buf, wait_for=[eventM1])


            # Step #11. Move the kernel's output data to host memory.
            matrix_dot_vector_copy_event0 = \
            cl.enqueue_copy(queues[0], srcRes[:halfSize*3,slicedSmp[iSliced]:slicedSmp[iSliced+1]], destination_buf,
                            is_blocking=False,
                            wait_for=[matrix_dot_vector_kernel_event0])
            matrix_dot_vector_copy_event1 = \
            cl.enqueue_copy(queues[1], srcRes[halfSize*3:,slicedSmp[iSliced]:slicedSmp[iSliced+1]], destination_buf,
                            is_blocking=False,
                            wait_for=[matrix_dot_vector_kernel_event1],
                            device_offset=halfSize*3*nlocalSmp*8)


            cl.wait_for_events([matrix_dot_vector_copy_event0, matrix_dot_vector_copy_event1])


    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))

    print(srcRes)
