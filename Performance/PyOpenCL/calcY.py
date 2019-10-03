import numpy as np
import sys

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
    nSmp = 1000
    M = np.tile(M, (1, nSmp, 1, 1))
    v = np.ones((m*3, nSmp))
    y = np.zeros((m*3, nSmp))

    for row in range(m):
        for i in range(indptr[row], indptr[row+1]):
            for s in range(nSmp):
                y[row*3:(row+1)*3, s] += np.dot(M[i, s, :, :], v[indices[i]*3:(indices[i]+1)*3, s])

    np.save('y', y)
