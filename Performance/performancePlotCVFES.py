# CPU final plot
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

# x: total time consuming executing nSmp
# y: nSmp * time consuming executing 1 sample
class PlotData:
    def __init__(self, name, x, nSmps, marker=None, color=None):
        self.name = name
        self.nSmps = nSmps[:len(x)]
        self.x = x
        self.y = x[0] * self.nSmps
        self.marker = marker
        self.color = color


# Plot the Performance rate btw actual time and time of 1 sample * nSmp.
def PPlot(plotDataArray, title, plotName, xlim, ylim, ibase=0):

    fs=8
    ms=6
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # Figure
    plt.figure(figsize=(3,3))
    plt.plot(plotDataArray[ibase].x, plotDataArray[ibase].x, '-.', c='k', label='Equal Performance')

    ax = plt.gca()
    for plotData in plotDataArray:
        # color = next(ax._get_lines.prop_cycler)['color']
        # # color = ax._get_lines.color_cycle.next()
        # Scatter the corresponding value.
        plt.plot(plotData.x, plotData.y, plotData.marker, ms=ms, c=plotData.color, label=plotData.name)
        # Fit the line.
        z = np.polyfit(plotData.x, plotData.y, 1)
        p = np.poly1d(z)
        x = np.linspace(plotData.x[0], plotDataArray[ibase].x[-1], 100)
        plt.plot(x, p(x), '--', c=plotData.color)

    plt.xlabel(r'CPU Time for $n$ Samples [min]', fontsize=fs)
    plt.ylabel(r'CPU Time for 1 Sample $\times n$ [min]', fontsize=fs)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,1))
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # plt.title('Performance Comparison of Mesh', fontsize=fs)
    plt.title(title, fontsize=fs)
    # plt.legend(loc='upper left', fontsize=fs)
    plt.legend(fontsize=fs, handlelength=3)
    # plt.grid(True, which='both', alpha=0.2)
    # plt.xlim([0.0, plotDataArray[ibase].x[-1]+1.0])
    # plt.ylim([-100.0, 1000.0])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(plotName)


def TPlot(plotDataArray, title, plotName, ibase=0):
    fs=8
    ms=6
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # Figure
    plt.figure(figsize=(3,3))

    ax = plt.gca()
    for plotData in plotDataArray:
        color = next(ax._get_lines.prop_cycler)['color']
        # color = ax._get_lines.color_cycle.next()
        # Scatter the corresponding value.
        plt.scatter(plotData.nSmps, plotData.x, marker='s', s=ms, c=color)
        # Fit the line.
        z = np.polyfit(plotData.nSmps, plotData.x, 1)
        p = np.poly1d(z)
        x = np.linspace(plotData.nSmps[0], plotDataArray[ibase].nSmps[-1], 100)
        plt.plot(x, p(x), label=plotData.name, c=color)

    plt.xlabel('Number of samples', fontsize=fs)
    plt.ylabel('Computing time (min)', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # plt.title('Performance Comparison of Mesh', fontsize=fs)
    plt.title(title, fontsize=fs)
    plt.legend(loc='upper left', fontsize=fs)
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    plt.savefig(plotName)


def SpeedUp(plotDataArray, nSmps):

    smpRate = []
    for plotData in plotDataArray:
        smpRate.append(plotData.y/plotData.x)

    puRate = np.zeros((len(plotDataArray), len(nSmps)))
    for iSmp in range(len(nSmps)):
        ibase = 0
        for iPU in range(len(plotDataArray)):
            if not len(plotDataArray[iPU].x) >= iSmp + 1:
                ibase += 1
                continue

            puRate[iPU, iSmp] = plotDataArray[ibase].x[iSmp]/plotDataArray[iPU].x[iSmp]

    print('smpRate\n {}\npuRate\n {}'.format(smpRate, puRate))

def TimeFormatting(CPUTimes):
    mytimes = [[str(datetime.timedelta(seconds=int(t*60))) for t in tpu] for tpu in CPUTimes]
    # mytimes.extend([[str(datetime.timedelta(seconds=int(t*60))) for t in tpu] for tpu in GPUTimes])
    print(mytimes)


def CoarseMeshPlot(Markers, Colors):
    # Sparse Mesh
    nSmps = np.array([1, 10, 50, 100, 200, 500])
    GPUTimes = np.array([[48983.0, 55116.9, 102640.4, 187255.4, 330740.4, 759271.1], # 1 GPU
                         [52269.8, 53984.1, 82260.3, 142749.3, 223668.7, 505900.6], # 2 GPUs
                         #52085.6, 53664.9, 82502.4, 142729.1, 223920.4, 506236.4 rank 1 of 2 GPUs
                         [64634.6, 70172.8, 71487.3, 129271.9, 201338.9, 451216.8], # 3 GPUs
                         #64879.0, 69951.0, 71339.1, 129485.0, 201381.6, 451249.0
                         #64837.0, 70202.3, 71597.0, 129496.9, 201350.1, 450988.2
                         [60114.2, 56834.4, 79684.8, 122129.8, 205278.0, 453259.6]]) / 6.0e4 # 4 GPUs
                         #60420.1, 56603.4, 79924.0, 122138.9, 205019.1, 453061.6
                         #60424.0, 56717.2, 79888.8, 122118.6, 205303.5, 453169.7
                         #60381.1, 57016.4, 79931.7, 122113.2, 205304.1, 453247.5
    GPUNames = ['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']

    CPUTimes = np.array([[489779.8, 3424007.5, 17600008.1, 34008103.9, 67210810.6, 180408996.4],
                         [326579.8, 552005.0, 1829698.7, 3545064.7, 6764989.8, 17261157.7],
                         [208469.8, 531083.1, 2441160.6, 4144606.4, 9777574.6, 20941943.7]]) / 6.0e4
    CPUNames = ['1 CPU', '12 CPUs', '24 CPUs']

    plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps, Markers[i], Colors[i]) for i in range(len(CPUNames))]
    plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps, Markers[3+i], Colors[3+i]) for i in range(len(GPUNames))])
    # PPlot(plotDataArray, 'Mesh of 5074 cells 2565 nodes', 'PerformanceCVFES.pdf', ibase=3)
    # TPlot(plotDataArray, 'Mesh of 5074 cells 2565 nodes', 'ComputingTimeCVFES.pdf', ibase=3)
    # SpeedUp(plotDataArray, nSmps)

    xlim = [0.0, 80.0]
    ylim = [0.0, 4000.0]
    PPlot(plotDataArray, 'Scaling Results for Small Mesh', 'CoarsePerformGPUs.pdf', xlim, ylim, ibase=0)


def MediumMeshPlot(Markers, Colors):
    # Fine Mesh
    nSmps = np.array([1, 10, 50, 100, 200, 500])
    GPUTimes = np.array([[42002.5, 70975.8, 120105.6, 210177.3, 361385.2, 841459.0],
                         [59572.9, 61098.5, 106357.9, 172145.3, 344984.1, 731003.0],
                         #59341.3  61273.8  106127.0  172122.5  344972.7  730995.1
                         [63949.6, 65100.5, 114644.2, 168993.5, 334620.9, 709998.2],
                         #63667.4  65114.5  114971.9  169151.7  334652.9  709933.9
                         #64000.0  64841.8  114959.1  169111.2  334488.5  709563.5
                         [56289.1, 72885.3, 107319.6, 168513.0, 307534.6, 729937.3]]) / 6.0e4
                         #56058.4  72756.5  107227.2  168282.0  307560.9  730005.1
                         #56229.1  72829.3  107247.6  168498.7  307535.5  729915.5
                         #56183.3  72602.8  107272.0  168538.4  307552.2  729890.7
    GPUNames = ['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']
    CPUTimes = np.array([[1132922.2, 9734767.2, 46000219.5, 94498960.4, 222079906.0, 519162802.8],
                         [315593.9, 653607.3, 3329105.1, 7365001.1, 13305993.4, 34787364.0],
                         [341165.3, 847756.2, 3534784.9, 7151607.4, 13224270.9, 33563159.6]]) / 6.0e4
    CPUNames = ['1 CPU', '12 CPUs', '24 CPUs']

    plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps, Markers[i], Colors[i]) for i in range(len(CPUNames))]
    plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps, Markers[3+i], Colors[3+i]) for i in range(len(GPUNames))])
    # PPlot(plotDataArray, 'Mesh of 15136 cells 7628 nodes', 'PerformanceCVFES_FineMesh.pdf', ibase=3)
    # TPlot(plotDataArray, 'Mesh of 15136 cells 7628 nodes', 'ComputingTimeCVFES_FineMesh.pdf', ibase=3)
    # SpeedUp(plotDataArray, nSmps)

    xlim = [0.0, 80.0]
    ylim = [0.0, 4000.0]
    PPlot(plotDataArray, 'Scaling Results for Large Mesh', 'FinePerformGPUs.pdf', xlim, ylim, ibase=0)


def FineMeshPlot():
    # More Fine Mesh
    GPUTimes = np.array([[72233.6, 143849.5, 503000.8, 982297.4, 1936421.7, 0.0],
                         [75770.5, 153928.7, 492586.9, 985086.3, 1853315.8, 4549083.2],
                         #76382.6  153871.7  493229.7  985232.5  1853112.3  4550185.1
                         [81302.1, 193782.3, 492998.5, 938313.6, 1916629.4, 4618848.0],
                         #80841.2  193043.9  493545.2  938315.1  1916434.5  4617957.1
                         #81271.2  193874.1  493630.8  937710.9  1916125.3  4619006.8
                         [78998.1, 155279.7, 483981.7, 949364.1, 1840315.1, 5052295.2]])
                         #79870.7  154568.1  483963.3  949492.0  1839999.5  5051127.5
                         #79746.5  155170.2  483981.5  948950.1  1840385.8  5051715.9
                         #79598.2  155233.7  483278.6  949474.8  1840050.0  5051713.8
    GPUTimes = np.array([np.array(t) for t in GPUTimes]) / 6.0e4
    CPUTimes = np.array([[12236231.7, 102553162.2, 444475706.1, 855184337.7],
                         [675030.6, 3696468.2, 19770779.1, 38772651.9, 91506882.0],
                         [699123.2, 5315662.5, 24425963.3, 48351990.4, 98012682.9, 302023774.7]])
    CPUTimes = np.array([np.array(t) for t in CPUTimes]) / 6.0e4

    plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps) for i in range(len(CPUNames))]
    plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))])
    # PPlot(plotDataArray, 'Mesh of 131552 cells 65896 nodes', 'PerformanceCVFES_MoreFineMesh.pdf', ibase=3)
    # TPlot(plotDataArray, 'Mesh of 131552 cells 65896 nodes', 'ComputingTimeCVFES_MoreFineMesh.pdf', ibase=3)
    # SpeedUp(plotDataArray, nSmps)


# GPU cases rerun for the finest mesh. (Jan 5th 2021)
def GPUFinestMesh():
    nSmps = np.array([1, 10, 50, 100, 200, 500])
    # GPUTimes = np.array([[102774.4, 129356.9, 373549.0, 707558.8, 1342596.4, 0.0],
    #                      #
    #                      [59375.2, 86247.7, 384846.4, 546821.1, 1454883.3, 2474660.2],
    #                      [44581.6, 75284.1, 401674.6, 511108.7, 1457591.5, 2399783.0],
    #                      [40824.5, 65431.7, 363448.8, 469119.1, 1333384.4, 2218996.1],
    #                      ])

    # Jan 27th 2021
    GPUTimes = np.array([[29269.9, 30040.5, 60553.0, 120298.3, 213049.3, 0.0],
                         [16023.4, 16100.6, 31746.6, 62739.9, 113191.6, 258522.4],
                         [11005.1, 11363.8, 21804.6, 42832.9, 77033.7, 174785.7],
                         [8792.4, 9235.8, 17357.5, 33221.2, 59660.4, 136374.4]]) # 4 GPUs
    GPUTimes = np.array([np.array(t) for t in GPUTimes]) / 6.0e4
    GPUNames = ['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']

    CPUTimes = np.array([[12236231.7, 102553162.2, 444475706.1, 855184337.7, 0.0, 0.0],
                         [675030.6, 3696468.2, 19770779.1, 38772651.9, 91506882.0, 0.0],
                         [699123.2, 5315662.5, 24425963.3, 48351990.4, 98012682.9, 302023774.7]])
    # calc 1000 time steps from 30,000 time steps' time
    CPUTimes = np.array([np.array(t) for t in CPUTimes]) / 30.0 / 6.0e4
    CPUNames = ['1 CPU', '12 CPUs', '24 CPUs']

    # plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps) for i in range(len(CPUNames))]
    # plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))])
    # SpeedUp(plotDataArray, nSmps)
    TimeFormatting(GPUTimes)
    TimeFormatting(CPUTimes)

    for j in range(4):
        print('({},{})'.format(CPUTimes[0,0]*nSmps[j]/CPUTimes[0,j], CPUTimes[0,j]/CPUTimes[0,j]))
    for j in range(5):
        if j != 4:
            print('({},{})'.format(CPUTimes[1,0]*nSmps[j]/CPUTimes[1,j], CPUTimes[0,j]/CPUTimes[1,j]))
        else:
            print('({},)'.format(CPUTimes[1,0]*nSmps[j]/CPUTimes[1,j]))
    for j in range(6):
        if j == 5:
            print('({},)'.format(CPUTimes[2,0]*nSmps[j]/CPUTimes[2,j]))
        elif j == 4:
            print('({},{})'.format(CPUTimes[2,0]*nSmps[j]/CPUTimes[2,j], CPUTimes[1,j]/CPUTimes[2,j]))
        else:
            print('({},{})'.format(CPUTimes[2,0]*nSmps[j]/CPUTimes[2,j], CPUTimes[0,j]/CPUTimes[2,j]))



    for i in range(4):
        for j in range(6):
            if j == 5 and i == 0:
                break

            if j == 4:
                print('({},{})'.format(GPUTimes[i,0]*nSmps[j]/GPUTimes[i,j], CPUTimes[1][4]/GPUTimes[i,j]))
            elif j == 5:
                print('({},{})'.format(GPUTimes[i,0]*nSmps[j]/GPUTimes[i,j], CPUTimes[2][5]/GPUTimes[i,j]))
            else:
                print('({},{})'.format(GPUTimes[i,0]*nSmps[j]/GPUTimes[i,j], CPUTimes[0][j]/GPUTimes[i,j]))


# GPU cases rerun for the finest mesh. (Jan 27th 2021)
def GPUCoarseMesh():
    nSmps = np.array([1, 10, 50, 100, 200, 500])
    GPUTimes = np.array([[1732.4, 1698.6, 2904.8, 5289.7, 9156.5, 20470.8],
                         [1545.8, 1686.1, 2131.2, 3512.2, 5534.1, 11581.0],
                         [1720.4, 1804.4, 1917.7, 2883.6, 4570.2, 9375.4],
                         [1748.5, 1801.7, 1895.4, 2980.4, 4656.9, 9691.7], # 4 GPUs
                         ])
    GPUTimes = np.array([np.array(t) for t in GPUTimes]) / 6.0e4
    GPUNames = ['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']

    CPUTimes = np.array([[489779.8, 3424007.5, 17600008.1, 34008103.9, 67210810.6, 180408996.4],
                         [326579.8, 552005.0, 1829698.7, 3545064.7, 6764989.8, 17261157.7],
                         [208469.8, 531083.1, 2441160.6, 4144606.4, 9777574.6, 20941943.7]]) / 30.0 / 6.0e4
    CPUNames = ['1 CPU', '12 CPUs', '24 CPUs']

    TimeFormatting(GPUTimes)
    TimeFormatting(CPUTimes)

    for i in range(3):
        for j in range(6):
            print('({},{})'.format(CPUTimes[i,0]*nSmps[j]/CPUTimes[i,j], CPUTimes[0][j]/CPUTimes[i,j]))

    for i in range(4):
        for j in range(6):
            print('({},{})'.format(GPUTimes[i,0]*nSmps[j]/GPUTimes[i,j], CPUTimes[0][j]/GPUTimes[i,j]))


# GPU cases rerun for the finest mesh. (Jan 27th 2021)
def GPUFineMesh():
    nSmps = np.array([1, 10, 50, 100, 200, 500])
    GPUTimes = np.array([[3886.8, 3824.9, 7316.1, 13938.6, 24903.4, 56686.9],
                         [2515.9, 2487.5, 4312.0, 8131.0, 13526.5, 30831.3],
                         [2219.4, 2343.6, 3777.9, 6611.2, 10991.3, 24910.5],
                         [1887.7, 2298.9, 3261.9, 5371.1, 8984.6, 19421.2], # 4 GPUs
                         ])
    GPUTimes = np.array([np.array(t) for t in GPUTimes]) / 6.0e4
    GPUNames = ['1 GPU', '2 GPUs', '3 GPUs', '4 GPUs']
    CPUTimes = np.array([[1132922.2, 9734767.2, 46000219.5, 94498960.4, 222079906.0, 519162802.8],
                         [315593.9, 653607.3, 3329105.1, 7365001.1, 13305993.4, 34787364.0],
                         [341165.3, 847756.2, 3534784.9, 7151607.4, 13224270.9, 33563159.6]]) / 30.0 / 6.0e4
    CPUNames = ['1 CPU', '12 CPUs', '24 CPUs']

    TimeFormatting(GPUTimes)
    TimeFormatting(CPUTimes)

    for i in range(3):
        for j in range(6):
            print('({},{})'.format(CPUTimes[i,0]*nSmps[j]/CPUTimes[i,j], CPUTimes[0][j]/CPUTimes[i,j]))

    for i in range(4):
        for j in range(6):
            print('({},{})'.format(GPUTimes[i,0]*nSmps[j]/GPUTimes[i,j], CPUTimes[0][j]/GPUTimes[i,j]))



# 1 GPU, 3000 time steps, corse mesh
# 1 sample  10 samples  50 samples  100 samples     200         500
# 10049.5   9737.8      16729.3     29245.4         56119.6     114299.3
#           10.32       30.04       34.36           35.81       43.96


# 4 GPU, 3000 time steps, old finest mesh (no sync)
# 1 sample  10 samples  50 samples  100 samples     200         500
# 54410.8   60652.3     234465.8    452266.6        901203.2    2144025.8   2068119.4   2123402.2
#           8.97        11.6        12.03           12.08       12.69       13.15

# no sync and no update copying to GPU
# 42915.4 + 31790.6 + 47813.8 + 46653.6
# 46653.6   54650.0     226417.2    1943384.0 (500)
#           8.54        10.3        12.0

# # Corse mesh, 5074 elements, 2565 nodes, 1000 time steps of dt=1.0e-5
# [['0:00:16', '0:01:54', '0:09:46', '0:18:53', '0:37:20', '1:40:13'], 
#  ['0:00:10', '0:00:18', '0:01:00', '0:01:58', '0:03:45', '0:09:35'], 
#  ['0:00:06', '0:00:17', '0:01:21', '0:02:18', '0:05:25', '0:11:38']] # CPU

# [['0:00:01', '0:00:01', '0:00:02', '0:00:05', '0:00:09', '0:00:20'], 
#  ['0:00:01', '0:00:01', '0:00:02', '0:00:03', '0:00:05', '0:00:11'], 
#  ['0:00:01', '0:00:01', '0:00:01', '0:00:02', '0:00:04', '0:00:09'], 
#  ['0:00:01', '0:00:01', '0:00:01', '0:00:02', '0:00:04', '0:00:09']] # GPU

# (1.0,1.0) (1.43,1.0) (1.39,1.0) (1.44,1.0) (1.46,1.0) (1.36,1.0)
# (1.0,1.50) (5.92,6.20) (8.92,9.62) (9.21,9.59) (9.65,9.94) (9.46,10.45)
# (1.0,2.35) (3.93,6.45) (4.27,7.22) (5.03,8.21) (4.26,6.87) (4.98,8.61)

# (1.0,9.42) (10.20,67.19) (29.82,201.96) (32.75,214.30) (37.84,244.67) (42.31,293.77)
# (1.0,10.56) (9.17,67.69) (36.27,275.28) (44.01,322.76) (55.86,404.83) (66.74,519.27)
# (1.0,9.49) (9.53,63.25) (44.86,305.92) (59.66,393.12) (75.29,490.21) (91.75,641.43)
# (1.0,9.34) (9.70,63.35) (46.12,309.52) (58.67,380.35) (75.09,481.08) (90.21,620.49)


# # Fine mesh, 15136 elements, 7628 nodes, 1000 time steps of dt=1.0e-5
# [['0:00:37', '0:05:24', '0:25:33', '0:52:29', '2:03:22', '4:48:25'], 
#  ['0:00:10', '0:00:21', '0:01:50', '0:04:05', '0:07:23', '0:19:19'], 
#   ['0:00:11', '0:00:28', '0:01:57', '0:03:58', '0:07:20', '0:18:38']]

# [['0:00:03', '0:00:03', '0:00:07', '0:00:13', '0:00:24', '0:00:56'], 
#  ['0:00:02', '0:00:02', '0:00:04', '0:00:08', '0:00:13', '0:00:30'], 
#  ['0:00:02', '0:00:02', '0:00:03', '0:00:06', '0:00:10', '0:00:24'], 
#  ['0:00:01', '0:00:02', '0:00:03', '0:00:05', '0:00:08', '0:00:19']]

# (1.0,1.0) (1.16,1.0) (1.23,1.0) (1.20,1.0) (1.02,1.0) (1.09,1.0)
# (1.0,3.59) (4.83,14.89) (4.74,13.82) (4.29,12.83) (4.74,16.69) (4.54,14.92)
# (1.0,3.32) (4.02,11.48) (4.83,13.01) (4.77,13.21) (5.16,16.79) (5.08,15.47)

# (1.0,9.72) (10.16,84.84) (26.56,209.58) (27.89,225.99) (31.22,297.26) (34.28,305.28)
# (1.0,15.01) (10.11,130.45) (29.17,355.60) (30.94,387.40) (37.20,547.27) (40.80,561.29)
# (1.0,17.02) (9.47,138.46) (29.37,405.87) (33.57,476.46) (40.38,673.50) (44.55,694.70)
# (1.0,20.01) (8.21,141.15) (28.94,470.08) (35.15,586.47) (42.02,823.93) (48.60,891.06)

# # Finest mesh, 131552 elements, 65896 nodes, 1000 time steps of dt=1.0e-5
# [['0:06:47', '0:56:58', '4:06:55', '7:55:06', '0:00:00', '0:00:00'],
#  ['0:00:22', '0:02:03', '0:10:59', '0:21:32', '0:50:50', '0:00:00'], 
#  ['0:00:23', '0:02:57', '0:13:34', '0:26:51', '0:54:27', '2:47:47']]

# [['0:00:29', '0:00:30', '0:01:00', '0:02:00', '0:03:33', '0:00:00'], 
#  ['0:00:16', '0:00:16', '0:00:31', '0:01:02', '0:01:53', '0:04:18'], 
#  ['0:00:11', '0:00:11', '0:00:21', '0:00:42', '0:01:17', '0:02:54'], 
#  ['0:00:08', '0:00:09', '0:00:17', '0:00:33', '0:00:59', '0:02:16']]

# (1.0,1.0)    (1.19,1.0)   (1.38,1.0)     (1.43,1.0)
# (1.0,18.13)  (1.83,27.74) (1.71,22.48)   (1.74,22.06)   (1.46,1.0)
# (1.0,17.50)  (1.32,19.29) (1.43,18.20)   (1.45,17.69)   (1.43,0.93)   (1.16,1.0)

# (1.0,13.93) (9.74,113.79) (24.17,244.68) (24.33,236.96) (27.48,14.32)
# (1.0,25.45) (9.95,212.32) (25.24,466.69) (25.54,454.35) (28.31,26.95) (30.99,38.94)
# (1.0,37.06) (9.68,300.82) (25.24,679.48) (25.69,665.52) (28.57,39.60) (31.48,57.60)
# (1.0,46.39) (9.52,370.13) (25.33,853.57) (26.47,858.07) (29.47,51.13) (32.24,73.82)



if __name__ == "__main__":

    Markers = ['o', 'D', '^', '<', '*', '+', 'P']
    Colors = ['b', 'r', 'm', 'k', 'g', 'y', 'c']

    # CoarseMeshPlot(Markers, Colors)
    # MediumMeshPlot(Markers, Colors)

    GPUCoarseMesh()
    GPUFineMesh()
    GPUFinestMesh()










