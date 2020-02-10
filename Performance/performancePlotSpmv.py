# CPU final plot
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot(savefilename, x, ys, marks, labels, xylabel, xylim=None, title=None):

    fs=8
    ms=6
    plt.rc('font',  family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text',  usetex=True)

    plt.figure(figsize=(3,3))
    for i,y in enumerate(ys):
        plt.semilogx(x, y, marks[i], label=labels[i], markersize=ms)
    plt.xlabel(xylabel[0], fontsize=fs)
    plt.ylabel(xylabel[1], fontsize=fs)
    plt.tick_params(labelsize=fs)
    if title is not None:
        plt.title(title, fontsize=fs)
    plt.legend(fontsize=fs-2) # ,loc='upper right')
    plt.grid(True, which='both',alpha=0.2)
    if xylim is not None:
        plt.xlim(xylim[0])
        plt.ylim(xylim[1])
    plt.tight_layout()
    plt.savefig(savefilename)


def mymean(x):
    y = np.array([np.mean(xi) for xi in x])
    return y

def CCythonPlot():
    nnz = np.array([160587, 478404, 4147128, 5839218, 1743660])

    # Plot {Cython in CVFES} vs. {seperate C implementation} of matrix multiplication.
    # cvtime = np.array([[0.33881, 0.33672, 0.35745],
    #                    [1.04072, 0.94864, 1.01663, 0.94251],
    #                    [21.04706, 21.23391, 21.31450],
    #                    [25.24464, 23.73951, 23.07769],
    #                    [5.91816, 7.14310, 6.08398, 6.75449]])

    # ctime = np.array([[0.93752, 0.51309, 0.53090],
    #                   [0.81215, 0.81665],
    #                   [24.92515, 24.17198, 22.54576],
    #                   [33.34184, 32.81031, 33.24440, 30.01300],
    #                   [9.66164, 9.17328, 10.11726, 8.82071]])

    # Feb 5th 2020 Redo data
    cvtime = np.array([[0.28183, 0.24757, 0.28375],
                       [0.99590, 0.91361, 0.79533],
                       [17.16590, 17.14658, 19.72813],
                       [28.37629, 26.23727, 31.11859],
                       [6.86122, 6.50774, 6.69672]])

    ctime = np.array([[0.41569, 0.34229, 0.34754],
                      [0.79592, 0.82765, 0.73826],
                      [21.74308, 20.69131, 27.42497],
                      [32.81260, 29.51813, 30.86369],
                      [9.74603, 9.09914, 7.92005]])

    # Calc performance
    cvtime = mymean(cvtime)
    cvpfl = 2.0*nnz/cvtime*1000.0/1.0e9
    ctime = mymean(ctime)
    cpfl = 2.0*nnz/ctime*1000.0/1.0e9


    # Set up the plotting parameters.
    ys = np.array([cvpfl, cpfl])
    marks = ['*', '^']
    labels = ['Cython imp.', 'C imp.']
    xylabel = ['NNZ', 'Performance using 1 Thread (Gflop/s)']
    xylim = np.array([[1e5,1e7], [0.1,1.4]])

    plot('CCythonCmp.pdf', nnz, ys, marks, labels, xylabel, xylim)


def MKLCythonPlot():
    nnz = np.array([160587, 478404, 4147128, 5839218, 1743660])

    # Plot num_threads>1, {MKL} vs. {Cython+OpenMP}
    # cvtime = np.array([[0.33881, 0.33672, 0.35745],
    #                    [1.04072, 0.94864, 1.01663, 0.94251],
    #                    [21.04706, 21.23391, 21.31450],
    #                    [25.24464, 23.73951, 23.07769],
    #                    [5.91816, 7.14310, 6.08398, 6.75449]])

    # cvtimeth2 = np.array([[0.21193, 0.17754, 0.17855, 0.20822],
    #                       [0.44982, 0.45808, 0.53927, 0.49462],
    #                       [13.10864, 14.74172, 16.88791, 14.64024],
    #                       [19.48594, 17.70486, 13.12463, 19.82757, 16.54540],
    #                       [1.86667, 2.00633, 2.15453, 1.73916]])

    # cvtimeth3 = np.array([[0.15325, 0.11901, 0.15585, 0.14201],
    #                       [0.33546, 0.36989, 0.32643, 0.36780],
    #                       [10.29470, 10.68427, 9.01274, 11.89203],
    #                       [9.01016, 10.96982, 10.13198, 13.52591, 10.93790, 8.41403],
    #                       [1.25448, 1.17585, 1.46750, 1.31325]])


    # mkltime = np.array([[1.12609, 0.54260, 0.46984],
    #                     [0.74460, 0.74154],
    #                     [22.54826, 24.04722, 23.94690],
    #                     [34.79566, 31.41928, 31.76151, 29.16877],
    #                     [10.18332, 8.33233, 9.98194, 8.29829]])

    # mkltimeth2 = np.array([[0.68878, 0.12237, 0.16147, 0.16146, 0.13286],
    #                        [0.40101, 0.39988, 0.38557, 0.40748],
    #                        [14.64503, 14.34379, 11.86394, 10.81161],
    #                        [17.97887, 23.35952, 25.06998, 22.47638, 21.53759],
    #                        [3.66261, 4.84905, 4.50576, 4.16545]])

    # mkltimeth3 = np.array([[0.10296, 0.10448, 0.11266, 0.11891],
    #                        [0.32423, 0.29295, 0.29761, 0.28256],
    #                        [7.99240, 11.33886, 11.61397, 15.82518],
    #                        [12.66775, 14.86586, 13.95960, 15.03994],
    #                        [1.74590, 1.20944, 1.55321, 2.28035, 2.94365, 1.94694, 1.18920]])


    # Feb 5th 2020 Redo data
    cvtime = np.array([[0.28183, 0.24757, 0.28375],
                       [0.99590, 0.91361, 0.79533],
                       [17.16590, 17.14658, 19.72813],
                       [28.37629, 26.23727, 31.11859],
                       [6.86122, 6.50774, 6.69672]])

    cvtimeth2 = np.array([[0.15164, 0.18434, 0.22124],
                          [0.45970, 0.47226, 0.53104],
                          [11.65718, 13.11375, 10.66711],
                          [12.81185, 14.80205, 16.86431],
                          [3.10471, 3.07283, 3.18582]])

    cvtimeth3 = np.array([[0.15935, 0.13765, 0.13972],
                          [0.28875, 0.33063, 0.33387],
                          [8.44883, 6.12164, 7.52501],
                          [13.29912, 11.92998, 12.86458],
                          [1.55530, 1.66300, 1.95413]])

    mkltime = np.array([[0.27653, 0.28642, 0.29836],
                        [0.77075, 0.68340, 0.69161],
                        [22.53760, 21.80466, 25.91593],
                        [33.23767, 30.10688, 40.21116],
                        [8.38105, 8.13462, 7.51954]])

    mkltimeth2 = np.array([[0.16492, 0.16040, 0.16272],
                           [0.42213, 0.39896, 0.42823],
                           [10.19132, 11.06307, 10.34972],
                           [18.10704, 16.37498, 14.21832],
                           [5.51219, 3.40986, 3.63801]])

    mkltimeth3 = np.array([[0.12096, 0.13599, 0.11916],
                           [0.36034, 0.29462, 0.29383],
                           [7.06282, 10.62690, 11.35423],
                           [11.31129, 15.96177, 14.90747],
                           [2.02527, 3.65368, 2.05858]])


    # Calc performance
    cvtime = mymean(cvtime)
    cvpfl = 2.0*nnz/cvtime*1000.0/1.0e9
    cvtimeth2 = mymean(cvtimeth2)
    cvpflth2 = 2.0*nnz/cvtimeth2*1000.0/1.0e9
    cvtimeth3 = mymean(cvtimeth3)
    cvpflth3 = 2.0*nnz/cvtimeth3*1000.0/1.0e9
    mkltime = mymean(mkltime)
    mklpfl = 2.0*nnz/mkltime*1000.0/1.0e9
    mkltimeth2 = mymean(mkltimeth2)
    mklpflth2 = 2.0*nnz/mkltimeth2*1000.0/1.0e9
    mkltimeth3 = mymean(mkltimeth3)
    mklpflth3 = 2.0*nnz/mkltimeth3*1000.0/1.0e9

    # # Set up the plotting parameters.
    # ys = np.array([cvpfl, mklpfl, cvpflth2, mklpflth2, cvpflth3, mklpflth3])
    # marks = ['*', '^', 'o', 's', 'P', 'x']
    # labels = ['Cython+oMP 1t', 'MKL 1t', 'Cython+oMP 2t', 'MKL 2t', 'Cython+oMP 3t', 'MKL 3t']
    # xylabel = ['NNZ', 'Performance (Gflop/s)']
    # xylim = np.array([[1e5,1e7], [0.0,3.5]])
    # plot('MKLCythonCmp.pdf', nnz, ys, marks, labels, xylabel, xylim)


    # The time consuming of using PyOpenCL with 1 GPU
    # cltime = np.array([[1.76193, 1.97433, 2.05549], [4.04370, 4.62150, 4.33438], [22.69509, 25.44719, 25.36407, 25.91869], [41.59847, 41.80934, 41.37926, 41.68538], [11.37067, 12.17777, 11.28679, 12.31105]])
    # cltime = mymean(cltime)
    # clpfl = 2.0*nnz/cltime*1000.0/1.0e9

    cltotaltime = np.array([[1.44469, 1.36989, 1.39771],
                            [2.95072, 2.94842, 3.13835, 3.15456],
                            [20.45331, 20.35658,  19.72866],
                            [58.93508, 52.13633, 58.65376, 57.40077],
                            [8.72397, 8.84117, 8.95797]])

    clcoretime = np.array([[0.82221, 0.76171, 0.79206],
                           [1.52754, 1.49491, 1.59589, 1.60349],
                           [8.70770, 9.21152, 8.74659],
                           [15.45186, 14.40355, 15.49689, 14.30226],
                           [3.96215, 3.80561, 3.90539]])

    cltotaltime = mymean(cltotaltime)
    cltotalpfl = 2.0*nnz/cltotaltime*1000.0/1.0e9
    clcoretime = mymean(clcoretime)
    clcorepfl = 2.0*nnz/clcoretime*1000.0/1.0e9

    # Set up the plotting parameters.
    ys = np.array([cvpfl, mklpfl, cvpflth2, mklpflth2, cvpflth3, mklpflth3, cltotalpfl, clcorepfl])
    marks = ['*', '^', 'o', 's', 'P', 'x', '.', 'X']
    labels = ['Cython+oMP 1t', 'MKL 1t', 'Cython+oMP 2t', 'MKL 2t', 'Cython+oMP 3t', 'MKL 3t', '1GPU Total', '1GPU Core']
    xylabel = ['NNZ', 'Performance (Gflop/s)']
    xylim = np.array([[1e5,1e7], [0.0,4.0]])


    # 'Performance comparison btw MKL, Cython and GPU imp. executing with 1 sample'
    plot('MKLCythonCmp_GPU.pdf', nnz, ys, marks, labels, xylabel, xylim)


if __name__ == "__main__":

    # CCythonPlot()
    MKLCythonPlot()
