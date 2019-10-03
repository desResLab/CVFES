from __future__ import division
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix, diags, issparse, linalg as sla
from scipy.special import gamma
from math import pi
import numpy as np
import vtk
import timeit
import matplotlib.pyplot as plt
from scipy.special import kv

def matern_covariance(d, nu=1.0, k=1.0):
    var = gamma(nu) / (gamma(nu+1.0) * ((4*pi)**1.0) * k**(2*nu))
    # print var
    cov = 1.0 / (gamma(nu)*(2**(nu-1.0))) * ((k*d)**nu) * kv(nu,k*d)
    # print cov
    return cov

def check_correlation(X, npNodes, k):
    ptsidx = np.random.choice(np.arange(1, len(npNodes)), 100)
    corX = np.corrcoef(X)
    distance = np.sqrt(np.sum((npNodes[ptsidx] - npNodes[0]) ** 2, axis=1))
    plt.plot(distance, corX[0, ptsidx], 'bo', markersize=3.0, label='generation')
    plt.plot(distance, matern_covariance(distance, nu=2.0, k=k), 'ro', markersize=3.0, label='Matern') # nu=0.5 for 3-dim, 1.0 for 2-dim
    plt.ylabel('Correlation',fontsize=17)
    plt.xlabel('Distance',fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.legend(fontsize=17)
    plt.show()

def check_variance(X, dim, nu, k):
    # Variance of the realizations
    varX = np.var(X, axis=1)
    # Theoretically
    var = gamma(nu) / (gamma(nu+dim/2)*(4*pi)**(dim/2)*k**(2*nu))

    plt.plot(np.arange(len(X)), varX, 'bo', label='generation')
    plt.plot(np.arange(len(X)), var*np.ones(len(X)), label='theoretical')
    plt.legend()
    plt.show()

def loc(indptr, indices, i, j):
    return indptr[i] + np.where(indices[indptr[i]:indptr[i+1]]==j)[0]

def GMRF(filename, mu=0.0, sigma=1.0, rho=3.7, samplenum=100, resfilename=None):

    # kappa = 7.0
    # nu = 1.0
    # kappa = 0.8
    nu = 2.0
    dim = 2.0

    # kappa = (gamma(nu)/(gamma(nu+1.0)*(4.0*pi)*(sigma**2.0))) **(1.0/(2.0*nu))
    # print('kappa {}'.format(kappa))
    # print('covariance distance {}'.format((8.0*nu)**0.5 / kappa))

    kappa = ((8.0*nu)**0.5)/rho
    sigmaOrigin = (gamma(nu)/(gamma(nu+1.0)*(4.0*pi)*(kappa**(2.0*nu))))**0.5
    sigmaRatio = sigma/sigmaOrigin

    # start_time = timeit.default_timer()

    # Read triangulation from file.
    # print 'Reading File...'
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polyDataModel = reader.GetOutput()

    totalNodes = polyDataModel.GetNumberOfPoints()
    vtkNodes = polyDataModel.GetPoints().GetData()
    npNodes = vtk_to_numpy(vtkNodes)
    # print(vtkPointData)
    # print(npNodes)

    # print 'Total nodes: ', totalNodes
    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Building Topology...'
    totalElms = polyDataModel.GetNumberOfCells()
    # Get cells from source file.
    npElms = np.zeros((totalElms, 3), dtype=int)
    npEdges = np.zeros((totalElms, 3, 3))
    npAreas = np.zeros(totalElms)
    for icell in range(totalElms):
        cell = polyDataModel.GetCell(icell)
        numpts = cell.GetNumberOfPoints()
        for ipt in range(numpts):
            npElms[icell, ipt] = cell.GetPointId(ipt)
        npEdges[icell, 0] = npNodes[npElms[icell, 2]] - npNodes[npElms[icell, 1]]
        npEdges[icell, 1] = npNodes[npElms[icell, 0]] - npNodes[npElms[icell, 2]]
        npEdges[icell, 2] = npNodes[npElms[icell, 1]] - npNodes[npElms[icell, 0]]
        npAreas[icell] = cell.ComputeArea()
        # for iedge in range(numedges):
        #   edge = cell.GetEdge(iedge)

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Creating the sparse matrix...'
    # Create sparse data structure.
    sparseInfo = [[] for _ in range(totalNodes)]
    for icell in range(totalElms):
        for inode in npElms[icell]:
            # [sparseInfo[inode].extend([pt]) for pt in npElms[icell] if pt not in sparseInfo[inode]]
            sparseInfo[inode].extend(npElms[icell])
    sparseInfo = np.array(sparseInfo)
    for knodes in range(totalNodes):
        sparseInfo[knodes] = np.unique(sparseInfo[knodes])
    # print(sparseInfo)

    # Generate the sparse matrix.
    indptr = [0]
    indices = []
    for inode in range(totalNodes):
        indices.extend(sparseInfo[inode])
        indptr.extend([len(indices)])
    rawC = np.zeros(len(indices))
    rawG = np.zeros(len(indices))
    # rawCTuta = np.zeros(totalNodes)

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Assembling global matrix...'
    # Generate C and G matrix.
    cm = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    # dcm = np.array([1.0, 1.0, 1.0])
    for icell in range(totalElms):
        # Compute local matrix first.
        localc = cm * npAreas[icell] / 12.0
        # localdc = dcm * npAreas[icell] / 3.0
        localg = np.dot(npEdges[icell], npEdges[icell].transpose()) / (4 * npAreas[icell])
        # Assembly to the glabal matrix.
        for i in range(3):
            # rawCTuta[npElms[icell, i]] += localdc[i]
            for j in range(3):
                rawindex = loc(indptr, indices, npElms[icell, i], npElms[icell, j])
                rawC[rawindex] += localc[i, j]
                rawG[rawindex] += localg[i, j]

    C = csc_matrix((rawC, np.array(indices), np.array(indptr)), shape=(totalNodes, totalNodes))
    G = csc_matrix((rawG, np.array(indices), np.array(indptr)), shape=(totalNodes, totalNodes))

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Creating inverse C...'
    invCTuta = diags([1.0 / C.sum(axis=1).transpose()], [0], shape=(totalNodes, totalNodes))

    # print 'Computating C Inverse...'
    # factorC = cholesky(C)
    # invC = factorC.inv()

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # Compute Q matrix according to C and G.
    # print 'Computing K...'
    K = (kappa**2)*C + G
    # print K.todense()

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Computing of Q...'
    Q1 = K
    # Q = (K.dot(invC)).dot(K)
    # Q2 = (K.dot(invCTuta)).dot(K) # Q2

    # alpha = int(nu+dim/2.0)
    # if alpha % 2 == 1:
    #     Qi = 3
    #     while Qi <= alpha:
    #         Q1 = (((K.dot(invCTuta)).dot(Q1)).dot(invCTuta)).dot(K)
    #         Qi += 2
    #     Q = Q1
    # else:
    #     Qi = 4
    #     while Qi <= alpha:
    #         Q2 = (((K.dot(invCTuta)).dot(Q2)).dot(invCTuta)).dot(K)
    #         Qi += 2
    #     Q = Q2

    Q = (((K.dot(invCTuta)).dot(Q1)).dot(invCTuta)).dot(K)
    # Q = (((K.dot(invC)).dot(Q1)).dot(invC)).dot(K)

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Cholesky factor of Q...'
    # Decomposition.
    factorQ = cholesky(Q) # ordering_method="natural"
    # L = factorQ.L()
    # print(factorQ.L())
    # lu = sla.splu(Q)
    # print(lu.L)
    # -- Get the permutation --
    P = factorQ.P()
    PT = np.zeros(len(P), dtype=int)
    PT[P] = np.arange(len(P))

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Generating samples...'
    # Generate normal distrib random nums & combine.
    # Z = np.random.normal(size=(totalNodes, samplenum))
    Z = np.empty((totalNodes, samplenum))
    for i in range(samplenum):
        Z[:,i] = np.random.normal(size=totalNodes)
    # Z = Z * sigmaRatio

    # print timeit.default_timer() - start_time
    # start_time = timeit.default_timer()

    # print 'Solving upper triangular syms...'
    X = factorQ.solve_Lt(Z, use_LDLt_decomposition=False)
    X = X[PT]
    # print np.allclose(, Z)

    X = X*sigmaRatio + mu
    # print timeit.default_timer() - start_time

    # X[X<=0.0] = 0.01

    if resfilename is not None:
        # start_time = timeit.default_timer()

        # Store back the random field.
        # print 'Exporting data...'
        vtkPointData = polyDataModel.GetPointData()
        for itrade in range(min(samplenum, 100)): # X.shape[1] # !not exporting all data to save time
            scaler = numpy_to_vtk(np.ascontiguousarray(X[:,itrade]))
            scaler.SetName('RandomField ' + str(itrade+1))
            vtkPointData.AddArray(scaler)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(polyDataModel)
        writer.SetFileName('{}.vtp'.format(resfilename))
        writer.Write()

        np.save(resfilename, X)

        # print timeit.default_timer() - start_time

    # print 'Ploting...'
    # check_correlation(X, npNodes, kappa)
    # check_variance(X, 2, 1.0, kappa)
    # check_variance(X, 3, 0.5, kappa)

    return X

if __name__ == '__main__':
    # solidfile = 'Examples/CylinderProject/mesh-complete/mesh-complete.exterior.vtp'
    # resThickness = 'Examples/CylinderProject/WallProperties/numberOfSamples/thickness'
    # resE = 'Examples/CylinderProject/WallProperties/numberOfSamples/YoungsModulus'

    solidfile = 'Examples/CylinderProject/refine-mesh-complete/mesh-complete.exterior.vtp'
    resThickness = 'Examples/CylinderProject/RefineWallProperties/numberOfSamples/thickness'
    resE = 'Examples/CylinderProject/RefineWallProperties/numberOfSamples/YoungsModulus'

    rho = 7.2
    # for iSmp in range(1,101):
    #     GMRF(solidfile, mu=0.4, sigma=0.04, rho=rho, samplenum=iSmp, resfilename='{}{:03}'.format(resThickness, iSmp))
    #     GMRF(solidfile, mu=7.0e6, sigma=1.0e5, rho=rho, samplenum=iSmp, resfilename='{}{:03}'.format(resE, iSmp))

    # solidfile = 'Examples/CylinderProject/refine-mesh-complete/mesh-complete.exterior.vtp'
    solidfile = 'Examples/CylinderProject/mesh-complete/mesh-surfaces/wall.vtp'
    # resThickness = 'Examples/CylinderProject/WallProperties/thickness'
    resE = 'Examples/CylinderProject/WallProperties/YoungsModulus'

    rhos = np.array([0.95, 3.7, 7.2])
    for rho in rhos:
        # GMRF(solidfile, mu=0.4, sigma=0.04, rho=rho, samplenum=100, resfilename='{}{}'.format(resThickness, rho))
        GMRF(solidfile, mu=7.0e6, sigma=7.0e5, rho=rho, samplenum=100, resfilename='{}{}'.format(resE, rho))

    # resE = 'Examples/CylinderProject/WallProperties/TestYoungsModulus'
    # GMRF(solidfile, mu=7.0e6, sigma=1.0e6, rho=0.95, samplenum=100, resfilename='{}{}'.format(resE, 0.95))
