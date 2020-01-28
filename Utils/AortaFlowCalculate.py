import numpy as np
import matplotlib.pyplot as plt


def FlowCalc(stime, etime, dt, cycletime, eqn):

    timesteps = np.arange(stime, etime, dt)
    if timesteps[-1] != etime:
        timesteps = np.append(timesteps, etime)

    flow = np.zeros_like(timesteps)

    for i, ct in enumerate(timesteps):
        t = ct - int(ct/cycletime)*cycletime
        flow[i] = eval(eqn)

    return np.stack((timesteps, flow), axis=-1)


if __name__ == "__main__":

    # # points = np.array([[0.0, 1.5], [0.02, 2.0], [0.15, 14.0], [0.28, 2.0], [0.3, 1.5], [0.35, 0.5], [0.65, 2.0], [0.75, 1.5]])
    # points = np.array([[0.0, 1.5], [0.02, 2.0], [0.15, 14.0], [0.28, 2.0], [0.3, 1.5]])
    # x = points[:,0]
    # y = points[:,1]
    # coefficients = np.polyfit(x, y, 4)
    # poly = np.poly1d(coefficients)
    # print(coefficients, poly)
    # new_x = np.linspace(x[0], x[-1])
    # new_y = poly(new_x)
    # plt.plot(x, y, 'o', new_x, new_y)
    # plt.show()

    stime = 0.0
    etime = 1.5
    dt = 0.001
    cycletime = 0.75
    eqn = '2.759e4*t**4-1.655e4*t**3+2.548e3*t**2-1.9565e1*t+1.5 if t>=0.0 and t<0.28 else -25.0*(t-0.28)+2.0 if t<=0.3 else 1.5-20.0*(t-0.3) if t<=0.35 else 0.5+5.0*(t-0.35) if t<=0.65 else -5.0*(t-0.65)+2.0 if t<=0.75 else 1.5'

    flow = FlowCalc(stime, etime, dt, cycletime, eqn)
    np.savetxt('../cfg/cylinderSparseInlet.flow', flow)

    plt.plot(flow[:,0], flow[:,1])
    plt.show()
