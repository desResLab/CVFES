# CPU final plot
import sys
import numpy as np
import matplotlib.pyplot as plt

def mymean(x):
    y = np.array([np.mean(xi) for xi in x])
    return y

nnz = np.array([160587, 478404, 4147128, 5839218, 1743660])

# Plot the performance on mesh cylinder (nSmp = 1000)
consume_t = mymean(np.array([[1584.17517], [1518.34377], [1287.57282], [115.10219]]))/1000
consume_core_t = mymean(np.array([[57.45468], [30.89908], [27.17567], [30.89908]]))/1000

performance = 2.0*nnz[0]/consume_t*1000.0/1.0e9
performance_core = 2.0*nnz[0]/consume_core_t*1000.0/1.0e9

# CPU's performance (CythonCVFES)
cpuPerformance = 2.0*nnz[0]/np.mean([0.33881, 0.33672, 0.35745])*1000.0/1.0e9

objects = ('Straitforward', 'Way 2', 'Way 4', 'Overlapping')
n_groups = len(objects)

index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.8


fs=17
ms=6
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

# Figure 1
plt.figure(figsize=(3,3))
fig, ax = plt.subplots()
rects1 = plt.bar(index, performance, bar_width, alpha=opacity, color='b', label='Total')
rects2 = plt.bar(index + bar_width, performance_core, bar_width, alpha=opacity, color='g', label='Core')
# Draw a line of CPU performance as reference.
plt.axhline(y=cpuPerformance, ls='--', linewidth=0.3, color='r') # linewidth=0.7, color='k'
plt.xlabel('', fontsize=fs)
plt.ylabel('Performance (Gflop/s)', fontsize=fs)
plt.title('performance of different tryings', fontsize=fs)
plt.xticks(index + bar_width, objects)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs-2)
# plt.grid(True, which='both',alpha=0.2)
plt.tight_layout()
plt.savefig('developPathPerformance.pdf')
exit(0)
# ----------------------- Sliced Overlapping -------------------------------------
# Plot the performance on mesh cabg (nSmp = 100)
consume_t = mymean(np.array([[4567.81036], [762.57044], [446.33433]]))/100

performance = 2.0*nnz[3]/consume_t*1000.0/1.0e9

# CPU's performance (CythonCVFES)
cpuPerformance = 2.0*nnz[3]/np.mean([25.24464, 23.73951, 23.07769])*1000.0/1.0e9

objects = ('Way 0', 'Way 1', 'Way 3')
n_groups = len(objects)

index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.8


fs=10
ms=6
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

# Figure 1
plt.figure(figsize=(3,3))
# fig, ax = plt.subplots()
rects1 = plt.bar(index, performance, bar_width, alpha=opacity, color='b', label='Total')
# Draw a line of CPU performance as reference.
plt.axhline(y=cpuPerformance, ls='--', linewidth=0.3, color='r') # linewidth=0.7, color='k'
plt.xlabel('', fontsize=fs)
plt.ylabel('Performance (Gflop/s)', fontsize=fs)
plt.title('performance of sliced overlapping', fontsize=fs)
plt.xticks(index, objects)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=fs-2)
# plt.grid(True, which='both',alpha=0.2)
plt.tight_layout()
plt.savefig('slicedOverlappingPerformance.pdf')

