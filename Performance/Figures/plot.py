""" The performance comparison btw MKL and my C blank implementation without using openMP """
import numpy as np
import matplotlib.pyplot as plt

lensize = np.array([494, 662, 1086, 2003, 3948, 4884, 6001, 7102, 10974, 15439, 25503, 217918])
nnz = np.array([1080, 1568, 11578, 42943, 60882, 147631, 1137751, 88618, 219812, 133840, 583240, 5926171])
mkltime = np.array([0.05250, 0.05947, 0.94973, 0.07787, 0.11389, 0.22729, 4.83246, 0.15627, 0.41260, 0.33868, 0.91748, 32.23834])
mytime = np.array([0.00540, 0.00752, 0.01434, 0.07026, 0.10224, 0.23039, 5.11980, 0.14548, 0.42357, 0.30447, 0.96300, 32.64027])

mkltime2 = np.array([0.02422, 0.02118, 0.02762, 0.05309, 0.06863, 0.12711, 2.59404, 0.09353, 0.18896, 0.15117, 0.44868, 23.70798])
mkltime3 = np.array([0.01918, 0.15442, 0.02887, 0.04313, 0.05387, 0.09336, 1.69522, 0.07166, 0.14769, 0.14802, 0.37298, 11.78258])

mklPerformance = 2*nnz/mkltime*1000.0/1.0e9
myPerformance = 2*nnz/mytime*1000.0/1.0e9
mklPerformance2 = 2*nnz/mkltime2*1000.0/1.0e9
mklPerformance3 = 2*nnz/mkltime3*1000.0/1.0e9

plt.plot(lensize, mklPerformance, '*', label='MKL 1 thread')
plt.plot(lensize, mklPerformance2, '^', label='MKL 2 threads')
plt.plot(lensize, mklPerformance3, 'D', label='MKL 3 threads')
plt.plot(lensize, myPerformance, 'o', label='My 1 thread')
plt.xlabel('Number of Equations', fontsize=20)
plt.ylabel('Performance Gflop/s', fontsize=20)
# plt.title('Performance', fontsize=20)
plt.legend()
plt.grid(True)
plt.show()

plt.plot(nnz, mklPerformance, '*', label='MKL 1 thread')
plt.plot(nnz, mklPerformance2, '^', label='MKL 2 threads')
plt.plot(nnz, mklPerformance3, 'D', label='MKL 3 threads')
plt.plot(nnz, myPerformance, 'o', label='My 1 thread')
plt.xlabel('NNZ', fontsize=20)
plt.ylabel('Performance Gflop/s', fontsize=20)
# plt.title('Performance', fontsize=20)
plt.legend()
plt.grid(True)
plt.show()

