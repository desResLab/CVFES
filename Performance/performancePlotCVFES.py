# CPU final plot
import sys
import numpy as np
import matplotlib.pyplot as plt

# x: total time consuming executing nSmp
# y: nSmp * time consuming executing 1 sample
class PlotData:
    def __init__(self, name, x, nSmps):
        self.name = name
        self.x = x
        self.y = x[0] * nSmps[:len(x)]


def PPlot(plotDataArray, plotName):

    fs=8
    ms=6
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    # Figure
    plt.figure(figsize=(3,3))
    plt.plot(plotDataArray[0].x, plotDataArray[0].x, '--', label='1:1 Line')

    ax = plt.gca()
    for plotData in plotDataArray:
        color = next(ax._get_lines.prop_cycler)['color']
        # Scatter the corresponding value.
        plt.scatter(plotData.x, plotData.y, marker='s', s=ms, c=color)
        # Fit the line.
        z = np.polyfit(plotData.x, plotData.y, 2)
        p = np.poly1d(z)
        plt.plot(plotData.x, p(plotData.x), label=plotData.name, c=color)

    plt.xlabel('Computing time for n samples (min)', fontsize=fs)
    plt.ylabel('Computing time for 1 sample * n (min)', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.title('Performance Comparison of Mesh', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid(True, which='both', alpha=0.2)
    # plt.xlim([1e5,1e7])
    # plt.ylim([0.2,1.4])
    plt.tight_layout()
    plt.savefig(plotName)


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

    # CPUTimes = np.array([])

    plotDataArray = [PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))]
    PPlot(plotDataArray, 'PerformanceCVFES.pdf')


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

    plotDataArray = [PlotData(GPUNames[i], GPUTimes[i], nSmps) for i in range(len(GPUNames))]
    PPlot(plotDataArray, 'PerformanceCVFES_FineMesh.pdf')




