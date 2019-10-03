# CPU final plot
import sys
import numpy as np
import matplotlib.pyplot as plt

def mymean(x):
    y = np.array([np.mean(xi) for xi in x])
    return y

nnz = np.array([160587, 478404, 4147128, 5839218, 1743660])

# Plot {Cython in CVFES} vs. {seperate C implementation} of matrix multiplication.
cvtime = np.array([[0.33881, 0.33672, 0.35745], [1.04072, 0.94864, 1.01663, 0.94251], [21.04706, 21.23391, 21.31450], [25.24464, 23.73951, 23.07769], [5.91816, 7.14310, 6.08398, 6.75449]])
cvtime = mymean(cvtime)
ctime = np.array([[0.93752, 0.51309, 0.53090], [0.81215, 0.81665], [24.92515, 24.17198, 22.54576], [33.34184, 32.81031, 33.24440, 30.01300], [9.66164, 9.17328, 10.11726, 8.82071]])
ctime = mymean(ctime)

cvpfl = 2.0*nnz/cvtime*1000.0/1.0e9
cpfl = 2.0*nnz/ctime*1000.0/1.0e9


fs=8
ms=6
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

# Figure 1
plt.figure(figsize=(3,3))
plt.semilogx(nnz, cvpfl, '*', label='Cython imp.', markersize=ms)
plt.semilogx(nnz, cpfl, '^', label='C imp.', markersize=ms)
plt.xlabel('NNZ', fontsize=fs)
plt.ylabel('Performance using 1 Thread (Gflop/s)', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs-2)
plt.grid(True, which='both',alpha=0.2)
plt.xlim([1e5,1e7])
plt.ylim([0.2,1.4])
plt.tight_layout()
plt.savefig('fig1.pdf')

# Plot num_threads>1, {MKL} vs. {Cython+OpenMP}
cvtime = np.array([[0.33881, 0.33672, 0.35745], [1.04072, 0.94864, 1.01663, 0.94251], [21.04706, 21.23391, 21.31450], [25.24464, 23.73951, 23.07769], [5.91816, 7.14310, 6.08398, 6.75449]])
cvtime = mymean(cvtime)
cvpfl = 2.0*nnz/cvtime*1000.0/1.0e9
cvtimeth2 = np.array([[0.21193, 0.17754, 0.17855, 0.20822], [0.44982, 0.45808, 0.53927, 0.49462], [13.10864, 14.74172, 16.88791, 14.64024], [19.48594, 17.70486, 13.12463, 19.82757, 16.54540], [1.86667, 2.00633, 2.15453, 1.73916]])
cvtimeth2 = mymean(cvtimeth2)
cvpflth2 = 2.0*nnz/cvtimeth2*1000.0/1.0e9
cvtimeth3 = np.array([[0.15325, 0.11901, 0.15585, 0.14201], [0.33546, 0.36989, 0.32643, 0.36780], [10.29470, 10.68427, 9.01274, 11.89203], [9.01016, 10.96982, 10.13198, 13.52591, 10.93790, 8.41403], [1.25448, 1.17585, 1.46750, 1.31325]])
cvtimeth3 = mymean(cvtimeth3)
cvpflth3 = 2.0*nnz/cvtimeth3*1000.0/1.0e9

mkltime = np.array([[1.12609, 0.54260, 0.46984], [0.74460, 0.74154], [22.54826, 24.04722, 23.94690], [34.79566, 31.41928, 31.76151, 29.16877], [10.18332, 8.33233, 9.98194, 8.29829]])
mkltime = mymean(mkltime)
mklpfl = 2.0*nnz/mkltime*1000.0/1.0e9
mkltimeth2 = np.array([[0.68878, 0.12237, 0.16147, 0.16146, 0.13286], [0.40101, 0.39988, 0.38557, 0.40748], [14.64503, 14.34379, 11.86394, 10.81161], [17.97887, 23.35952, 25.06998, 22.47638, 21.53759], [3.66261, 4.84905, 4.50576, 4.16545]])
mkltimeth2 = mymean(mkltimeth2)
mklpflth2 = 2.0*nnz/mkltimeth2*1000.0/1.0e9
mkltimeth3 = np.array([[0.10296, 0.10448, 0.11266, 0.11891], [0.32423, 0.29295, 0.29761, 0.28256], [7.99240, 11.33886, 11.61397, 15.82518], [12.66775, 14.86586, 13.95960, 15.03994], [1.74590, 1.20944, 1.55321, 2.28035, 2.94365, 1.94694, 1.18920]])
mkltimeth3 = mymean(mkltimeth3)
mklpflth3 = 2.0*nnz/mkltimeth3*1000.0/1.0e9

# The time consuming of using PyOpenCL with 1 GPU
# cltime = np.array([[1.76193, 1.97433, 2.05549], [4.04370, 4.62150, 4.33438], [22.69509, 25.44719, 25.36407, 25.91869], [41.59847, 41.80934, 41.37926, 41.68538], [11.37067, 12.17777, 11.28679, 12.31105]])
# cltime = mymean(cltime)
# clpfl = 2.0*nnz/cltime*1000.0/1.0e9

cltotaltime = np.array([[1.44469, 1.36989, 1.39771], [2.95072, 2.94842, 3.13835, 3.15456], [20.45331, 20.35658,  19.72866], [58.93508, 52.13633, 58.65376, 57.40077], [8.72397, 8.84117, 8.95797]])
cltotaltime = mymean(cltotaltime)
cltotalpfl = 2.0*nnz/cltotaltime*1000.0/1.0e9

clcoretime = np.array([[0.82221, 0.76171, 0.79206], [1.52754, 1.49491, 1.59589, 1.60349], [8.70770, 9.21152, 8.74659], [15.45186, 14.40355, 15.49689, 14.30226], [3.96215, 3.80561, 3.90539]])
clcoretime = mymean(clcoretime)
clcorepfl = 2.0*nnz/clcoretime*1000.0/1.0e9

# Figure 2
plt.figure(figsize=(3,3))
plt.semilogx(nnz, cvpfl, '*', label='Cython+oMP 1t')
plt.semilogx(nnz, mklpfl, '^', label='MKL 1t')
plt.semilogx(nnz, cvpflth2, 'o', label='Cython+oMP 2t')
plt.semilogx(nnz, mklpflth2, 's', label='MKL 2t')
plt.semilogx(nnz, cvpflth3, '+', label='Cython+oMP 3t')
plt.semilogx(nnz, mklpflth3, 'D', label='MKL 3t')
# plt.semilogx(nnz, clpfl, '.', label='1GPU')
plt.semilogx(nnz, cltotalpfl, '.', label='1GPU Total')
plt.semilogx(nnz, clcorepfl, 'X', label='1GPU Core')
plt.xlabel('NNZ', fontsize=fs)
plt.ylabel('Performance (Gflop/s)', fontsize=fs)
plt.tick_params(labelsize=fs)
plt.xlim([1e5,1e7])
plt.ylim([0.0,4.0])
plt.legend(fontsize=fs-2,loc='upper right')
plt.grid(True, which='both',alpha=0.2)
plt.tight_layout()
# plt.show()
plt.savefig('fig2.pdf')
