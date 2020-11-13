import numpy as np
from math import sin, pi
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

    stime = 0.0
    etime = 1.4
    dt = 0.001
    cycletime = 0.7
    # eqn = '84.0*t**2-0.22 if t>=0.0 and t<=0.05 else 21.0*(t-0.14)**2-0.18 if t<=0.14 else -0.18-0.67*(t-0.14) if t<=0.2 else -0.22+2.0*(t-0.2) if t<=0.21 else 30.86*(t-0.3)**2-0.45 if t<=0.3 else -0.45+0.62*(t-0.3) if t<=0.67 else -0.22'
    # eqn = '0.21*sin(t*8.975979)-0.22'
    eqn = '-0.115+0.105*sin(pi*(t-0.025)/0.05) if t>=0.0 and t<0.05 else -0.115-0.105*sin(pi*(t-0.125)/0.15) if t<0.2 else -0.335-0.115*sin(pi*(t-0.25)/0.1) if t<0.3 else -0.335+0.115*sin(pi*(t-0.485)/0.37) if t<0.67 else -0.22'

    flow = FlowCalc(stime, etime, dt, cycletime, eqn)
    # np.savetxt('../cfg/lcSparseInlet.flow', flow, fmt='%1.4e')
    # np.savetxt('../cfg/lcSinInflow.flow', flow, fmt='%1.4e')
    # np.savetxt('../cfg/lcInflow.flow', flow, fmt='%1.4e')

    # flow = np.loadtxt('lcSparseInlet.flow')
    # print(flow[flow[:,0]==0.005, 1])
    
    plt.plot(flow[:,0], flow[:,1])
    plt.show()
    print(np.mean(flow[:,1])) # -0.2804761486081371
