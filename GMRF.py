from __future__ import division
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix, diags, issparse, linalg as sla
from scipy.special import gamma
from scipy.special import kv
from math import pi, sqrt
import numpy as np
import vtk
import timeit
import matplotlib.pyplot as plt
import os.path


gdim = 2.0
gnu = 2.0

def Dijkstra(nNodes, nodes, neighbors, nbdists, source=0):
    Q = list(range(nNodes))
    dist = np.full(nNodes, np.inf)
    dist[source] = 0.0

    while len(Q) > 0:
        u = Q[np.argmin(dist[Q])]
        Q.remove(u)
        for iv, v in enumerate(neighbors[u]):
            if v not in Q:
                continue
            alt = dist[u] + nbdists[u][iv]
            if alt < dist[v]:
                dist[v] = alt

    return dist


def matern_covariance(d, nu=1.0, k=1.0):
    var = gamma(nu) / (gamma(nu+gdim/2.0) * ((4*pi)**(gdim/2.0)) * k**(2*nu))
    # print var
    cov = 1.0 / (gamma(nu)*(2**(nu-1.0))) * ((k*d)**nu) * kv(nu,k*d)
    # cov = var / (gamma(nu)*(2**(nu-1.0))) * ((k*d)**nu) * kv(nu,k*d)
    # print cov
    return cov

def check_correlation(X, npNodes, k, dists, source=0):
    ptsidx = np.random.choice(np.arange(1, len(npNodes)), 100)
    corX = np.corrcoef(X)
    # cov = np.cov(X)
    distance = dists[ptsidx]
    plt.plot(distance, corX[source, ptsidx], 'bo', markersize=3.0, label='generation')
    plt.plot(distance, matern_covariance(distance, nu=gnu, k=k), 'ro', markersize=3.0, label='Matern') # nu=0.5 for 3-dim, 1.0 for 2-dim
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

def unique(xlist):
    unique_list = []
    for x in xlist:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def check(filename, geofilename, rho, nu=2.0):

    # Read triangulation from file.
    print('Reading File...')
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(geofilename)
    reader.Update()
    polyDataModel = reader.GetOutput()

    totalNodes = polyDataModel.GetNumberOfPoints()
    vtkNodes = polyDataModel.GetPoints().GetData()
    npNodes = vtk_to_numpy(vtkNodes)

    nElements = polyDataModel.GetNumberOfCells()
    elements = np.empty((nElements, 3), dtype=int)
    for iElm in range(nElements):
        vtkCell = polyDataModel.GetCell(iElm)
        for ipt in range(3):
            elements[iElm,ipt] = vtkCell.GetPointId(ipt)

    # Collect the neighbors information.
    neighbors = [[] for _ in range(totalNodes)]
    uniqueNbs = []
    nbdists = []
    for iElm in range(nElements):
        for iNode in elements[iElm]:
            neighbors[iNode].extend(elements[iElm])
    for iNode in range(totalNodes):
        uniqueNbs.append(unique(neighbors[iNode]).remove(iNode))
        nbdists.append(np.linalg.norm(npNodes[neighbors[iNode]]-npNodes[iNode], axis=1))

    # Prepare the distance information.
    dists = Dijkstra(totalNodes, npNodes, neighbors, nbdists)

    # Read the random field generated.
    X = np.load(filename)

    # Calculate kappa
    kappa = ((2.0*nu)**0.5)/rho

    print('Ploting...')
    check_correlation(X, npNodes, kappa, dists)
    # check_variance(X, 2.0, nu, kappa)


def readNoise(filename, samplenum):
    print('Generating samples...')
    if os.path.exists(filename):
        return np.load(filename)

    # Z = np.random.normal(size=(totalNodes, samplenum))
    # Z = np.empty((totalNodes, samplenum))
    # # for i in range(samplenum):
    # #     Z[:,i] = np.random.normal(size=totalNodes)
    # for i in range(totalNodes):
    #     Z[i,:] = np.random.normal(size=samplenum)
    Z = np.random.multivariate_normal(np.zeros(totalNodes), np.identity(totalNodes), samplenum).T
    np.save(filename, Z)
    return Z


def loc(indptr, indices, i, j):
    return indptr[i] + np.where(indices[indptr[i]:indptr[i+1]]==j)[0]

class GMRF:

    def __init__(self, filename, dim=2.0, nu=2.0):

        # start_time = timeit.default_timer()
        # Read triangulation from file.
        print('Reading File...')
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        polyDataModel = reader.GetOutput()

        totalNodes = polyDataModel.GetNumberOfPoints()
        vtkNodes = polyDataModel.GetPoints().GetData()
        npNodes = vtk_to_numpy(vtkNodes)

        # print timeit.default_timer() - start_time
        # start_time = timeit.default_timer()
        print('Building Topology...')
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

        # Create sparse data structure.
        print('Creating the sparse matrix...')
        sparseInfo = [[] for _ in range(totalNodes)]
        for icell in range(totalElms):
            for inode in npElms[icell]:
                # [sparseInfo[inode].extend([pt]) for pt in npElms[icell] if pt not in sparseInfo[inode]]
                sparseInfo[inode].extend(npElms[icell])
        sparseInfo = np.array(sparseInfo)
        for knodes in range(totalNodes):
            sparseInfo[knodes] = np.unique(sparseInfo[knodes])

        # Generate the sparse matrix.
        indptr = [0]
        indices = []
        for inode in range(totalNodes):
            indices.extend(sparseInfo[inode])
            indptr.append(len(indices))
        rawC = np.zeros(len(indices))
        rawG = np.zeros(len(indices))

        # print timeit.default_timer() - start_time
        # start_time = timeit.default_timer()

        # print 'Assembling global matrix...'
        # Generate C and G matrix.
        cm = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
        # dcm = np.array([1.0, 1.0, 1.0])
        for icell in range(totalElms):
            # Compute local matrix first.
            localc = cm * npAreas[icell]
            # localdc = dcm * npAreas[icell] / 3.0
            localg = np.dot(npEdges[icell], npEdges[icell].transpose()) / (4.0 * npAreas[icell])
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
        # Remember things need to remember.
        self.dim = dim
        self.nu = nu
        self.C = C
        self.G = G
        self.invCTuta = invCTuta

        self.polyDataModel = polyDataModel
        self.totalNodes = totalNodes
        self.npNodes = npNodes


    def setRho(self, rho=0.95):

        C = self.C
        G = self.G
        invCTuta = self.invCTuta
        nu = self.nu

        kappa = ((2.0*nu)**0.5)/rho
        # start_time = timeit.default_timer()

        # Compute Q matrix according to C and G.
        # print 'Computing K...'
        K = (kappa**2)*C + G

        # print timeit.default_timer() - start_time
        # start_time = timeit.default_timer()

        # print 'Computing of Q...'
        Q1 = K
        Q2 = (K.dot(invCTuta)).dot(K) # Q2
        Q = (((K.dot(invCTuta)).dot(Q1)).dot(invCTuta)).dot(K) # Q3
        # Q = (((K.dot(invCTuta)).dot(Q2)).dot(invCTuta)).dot(K) # Q4

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
        self.kappa = kappa
        self.factorQ = factorQ
        self.PT = PT


    def generate(self, mu, sigma, Z, resfilename=None, lb=None):

        samplenum = Z.shape[1]
        # start_time = timeit.default_timer()

        # print 'Solving upper triangular syms...'
        X = self.factorQ.solve_Lt(Z, use_LDLt_decomposition=False)
        X = X[self.PT]

        # # sigmaReal0 = np.std(X)
        # # sigmaReal1 = np.amax(np.std(X, axis=0))
        # # sigmaReal2 = np.amax(np.std(X, axis=1))
        # # sigmaReal3 = np.mean(np.std(X, axis=1))
        # # print(sigmaReal0, sigmaReal1, sigmaReal2, sigmaReal3)
        # # return
        # # sigmaReal = min(np.amax(np.std(X, axis=0)), np.amax(np.std(X, axis=1)))
        # # sigmaReal = (np.std(X) + np.amax(np.std(X, axis=1))) / 2.0
        # sigmaReal = np.std(X) + (np.amax(np.std(X, axis=1)) - np.std(X)) * 0.667
        # sigmaRatio = sigma/sigmaReal
        # X = X*sigmaRatio + mu
        # # print timeit.default_timer() - start_time

        sigmaReal = np.std(X, axis=1)
        sigmaRatio = sigma/sigmaReal
        X = X*sigmaRatio[:,np.newaxis] + mu

        # # X[X<=0.0] = 0.01
        # if lb is not None:
        #     X[X<lb] = lb

        if resfilename is not None:
            # start_time = timeit.default_timer()

            # Store back the random field.
            # print 'Exporting data...'
            vtkPointData = self.polyDataModel.GetPointData()
            for itrade in range(min(samplenum, 100)): # X.shape[1] # !not exporting all data to save time
                scaler = numpy_to_vtk(np.ascontiguousarray(X[:,itrade]))
                scaler.SetName('RandomField ' + str(itrade+1))
                vtkPointData.AddArray(scaler)

            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(self.polyDataModel)
            writer.SetFileName('{}.vtp'.format(resfilename))
            writer.Write()

            np.save(resfilename, X)

            # print timeit.default_timer() - start_time

        return X


def generateGFs():

    # solidfile = 'Examples/CylinderProject/refine-more-mesh-complete/walls_combined.vtp'
    # noisefile = 'Examples/CylinderProject/MoreRefineWallPropertiesTest/noise.npy'
    # totalNodes = 16557 #2565 #7628 #16557
    # resThickness = 'Examples/CylinderProject/MoreRefineWallPropertiesTest/cyThickness'
    # thick_mu = 0.4
    # thick_sigma = 0.04
    # thick_lb = 0.28
    # resE = 'Examples/CylinderProject/MoreRefineWallPropertiesTest/cyYoungsModulus'
    # E_mu = 7.0e6
    # E_sigma = 7.0e5


    solidfile = 'Examples/lc/mesh-complete-5layers/walls_combined.vtp'
    totalNodes = 32091 # 22581
    resThickness = 'Examples/lc/lc5LayersWallProperties/lcThickness'
    thick_mu = 0.075
    thick_sigma = 0.017
    thick_lb = 0.024
    resE = 'Examples/lc/lc5LayersWallProperties/lcYoungsModulus'
    E_mu = 1.15e7
    E_sigma = 1.7e6


    samplenum = 100
    # Generate normal distrib random nums & combine.
    # Z = readNoise(noisefile, samplenum)
    np.random.seed(23)
    Z = np.random.normal(size=(totalNodes, samplenum))

    rhos = np.array([0.95, 3.7, 7.2])
    gf = GMRF(solidfile)
    for rho in rhos:
        gf.setRho(rho)
        gf.generate(mu=thick_mu, sigma=thick_sigma, Z=Z, resfilename='{}{}'.format(resThickness, rho), lb=thick_lb)
        gf.generate(mu=E_mu, sigma=E_sigma, Z=Z, resfilename='{}{}'.format(resE, rho))


def checkGFs():

    solidfile = 'Examples/lc/mesh-complete-5layers/walls_combined.vtp'
    resThickness = 'Examples/lc/5LayersWallProperties/cyThickness'
    resE = 'Examples/lc/5LayersWallProperties/cyYoungsModulus'

    rhos = np.array([0.95, 3.7, 7.2])
    for rho in rhos:
        check('{}{}.npy'.format(resThickness, rho), solidfile, rho)


if __name__ == '__main__':

    generateGFs()
    # checkGFs()

