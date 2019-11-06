# CPU final plot
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

# x: total time consuming executing nSmp
# y: nSmp * time consuming executing 1 sample
class PlotData:
    def __init__(self, name, x, nSmps):
        self.name = name
        self.nSmps = nSmps[:len(x)]
        self.x = x
        self.y = x[0] * self.nSmps


# Plot the Performance rate btw actual time and time of 1 sample * nSmp.
def PPlot(plotDataArray, title, plotName, ibase=0):

    fs=8
    ms=6
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # Figure
    plt.figure(figsize=(3,3))
    plt.plot(plotDataArray[ibase].x, plotDataArray[ibase].x, '--', label='1:1 Line')

    ax = plt.gca()
    for plotData in plotDataArray:
        color = next(ax._get_lines.prop_cycler)['color']
        # color = ax._get_lines.color_cycle.next()
        # Scatter the corresponding value.
        plt.scatter(plotData.x, plotData.y, marker='s', s=ms, c=color)
        # Fit the line.
        z = np.polyfit(plotData.x, plotData.y, 1)
        p = np.poly1d(z)
        x = np.linspace(plotData.x[0], plotDataArray[ibase].x[-1], 100)
        plt.plot(x, p(x), label=plotData.name, c=color)

    plt.xlabel('Computing time for n samples (min)', fontsize=fs)
    plt.ylabel('Computing time for 1 sample * n (min)', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # plt.title('Performance Comparison of Mesh', fontsize=fs)
    plt.title(title, fontsize=fs)
    plt.legend(loc='upper left', fontsize=fs)
    plt.grid(True, which='both', alpha=0.2)
    plt.xlim([0.0, plotDataArray[ibase].x[-1]+1.0])
    plt.ylim([-100.0, 1000.0])
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


if __name__ == "__main__":

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

    plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps) for i in range(len(CPUNames))]
    plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))])
    # PPlot(plotDataArray, 'Mesh of 5074 cells 2565 nodes', 'PerformanceCVFES.pdf', ibase=3)
    # TPlot(plotDataArray, 'Mesh of 5074 cells 2565 nodes', 'ComputingTimeCVFES.pdf', ibase=3)
    # SpeedUp(plotDataArray, nSmps)


    # Fine Mesh
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
    CPUTimes = np.array([[1132922.2, 9734767.2, 46000219.5, 94498960.4, 222079906.0, 519162802.8],
                         [315593.9, 653607.3, 3329105.1, 7365001.1, 13305993.4, 34787364.0],
                         [341165.3, 847756.2, 3534784.9, 7151607.4, 13224270.9, 33563159.6]]) / 6.0e4

    plotDataArray = [PlotData(CPUNames[i], CPUTimes[i], nSmps) for i in range(len(CPUNames))]
    plotDataArray.extend([PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))])
    # PPlot(plotDataArray, 'Mesh of 15136 cells 7628 nodes', 'PerformanceCVFES_FineMesh.pdf', ibase=3)
    # TPlot(plotDataArray, 'Mesh of 15136 cells 7628 nodes', 'ComputingTimeCVFES_FineMesh.pdf', ibase=3)
    # SpeedUp(plotDataArray, nSmps)


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


    mytimes = [[str(datetime.timedelta(seconds=int(t*60))) for t in tpu] for tpu in CPUTimes]
    # mytimes.extend([[str(datetime.timedelta(seconds=int(t*60))) for t in tpu] for tpu in GPUTimes])
    print(mytimes)
    exit(0)


