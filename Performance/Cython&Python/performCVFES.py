# Computing with the CVFES origin data structure.

import sys
import numpy as np
from timeit import default_timer as timer
from cy_spmv import MultiplyByVectorOrigin

LOOP_COUNT = 1000

def main():

    data = np.load(sys.argv[1])
    indptr = data['indptr']
    indices = data['indices']
    M = data['M']

    if len(sys.argv) > 2:
        M = np.tile(M, (1, int(sys.argv[2]), 1, 1))

    m = len(indptr)-1
    nSmp = M.shape[1]
    v = np.ones((m*3, nSmp))
    y = np.zeros((m*3, nSmp))

    start = timer()

    for i in range(LOOP_COUNT):
        MultiplyByVectorOrigin(indptr, indices, M, v, y)

    end = timer()

    print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit()

    main()
