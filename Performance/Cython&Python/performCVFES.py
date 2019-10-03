# Computing with the CVFES origin data structure.

import numpy as np
from timeit import default_timer as timer

from cy_spmv import MultiplyByVectorOrigin

LOOP_COUNT = 1000

data = np.load('cylinder.npz')

# print('{} \t {}'.format(len(X.indptr)-1, len(X.indices)))

m = len(data.indptr)-1
nSmp = data.M.shape[1]
v = np.ones((m*3, nSmp))
y = np.zeros((m*3, nSmp))

start = timer()

for i in range(LOOP_COUNT):
    MultiplyByVectorOrigin(data.indptr, data.indices, data.M, v, y)

end = timer()

print('OK, \t\t\t time: {:10.1f} ms'.format((end - start)/float(LOOP_COUNT) * 1000.0))
