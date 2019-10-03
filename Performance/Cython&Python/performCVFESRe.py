# Check the difference of using different data set every loop.
# It did slow down the computation somehow.

import numpy as np
from timeit import default_timer as timer

from cy_spmv import MultiplyByVectorOrigin

LOOP_COUNT = 1000

data = np.load('cylinder.npz')

# print('{} \t {}'.format(len(X.indptr)-1, len(X.indices)))

indptr = data['indptr']
indices = data['indices']
M = data['M']

m = len(indptr)-1
nSmp = M.shape[1]
v = np.ones((m*3, nSmp))
y = np.zeros((m*3, nSmp))

A = np.array([np.random.rand(M.shape[0], M.shape[1], M.shape[2], M.shape[3]) for i in range(LOOP_COUNT)])

start = timer()

for i in range(LOOP_COUNT):
    MultiplyByVectorOrigin(indptr, indices, A[i], v, y)

end = timer()

print('OK, \t\t\t time: {:10.5f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))
