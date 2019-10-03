# Pure cython spMV implementation with expended matrix.

import numpy as np
# Read in Matrix Market file
from scipy import io
from scipy.sparse import csr_matrix
from timeit import default_timer as timer

from cy_spmv import MultiplyByVector

LOOP_COUNT = 1000

X = io.mmread('moreFineCylinder.mtx')
X = csr_matrix(X)

# print('{} \t {}'.format(len(X.indptr)-1, len(X.indices)))

m = len(X.indptr)-1
v = np.ones(m)
y = np.zeros(m)

start = timer()

for i in range(LOOP_COUNT):
    MultiplyByVector(m, X.indptr, X.indices, X.data, v, y)

end = timer()

print('OK, \t\t\t time: {:10.1f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))
