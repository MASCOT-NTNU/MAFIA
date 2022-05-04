import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky
from MAFIA.Simulation.Config.Config import FILEPATH


class spde:
    def __init__(self, model = 2, reduce = False):
        """Initialize model

        Args:
            model (int, optional): Which GMRF to load. Defaults to 2.
            reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
        """
        self.M = 45
        self.N = 45
        self.P = 11
        self.n = self.M*self.N*self.P
        self.Stot = sparse.eye(self.n).tocsc()
        self.define(model=model)
        if reduce:
                self.reduce()

    
    def reduce(self):
        """Reduces the grid to have 7 depth layers instead of 11.
        """
        tx,ty,tz = np.meshgrid(np.arange(45),np.arange(45),np.arange(7))
        tx = tx.flatten()
        ty = ty.flatten()
        tz = tz.flatten()
        ks = ty*self.M*self.P + tx*self.P + tz
        self.Q = self.Q[ks,:][:,ks]
        self.Q_fac = cholesky(self.Q)
        self.M = 45
        self.N = 45
        self.P = 7
        self.n = self.M*self.N*self.P
        self.mu = np.load(FILEPATH + 'models/prior_small.npy')
        self.Stot = sparse.eye(self.n).tocsc()


    # fix par for models
    def define(self, model = 2):
        """Define the GMRF model

        Args:
            model (int, optional): Which model to use 1 or 2. Defaults to 2.
        """
        if (model==1):
            tmp = np.load(FILEPATH + 'models/SINMOD-NAs.npy')
        elif (model==2):
            tmp = np.load(FILEPATH + 'models/SINMOD-NAf.npy')
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
        self.Q_fac = cholesky(self.Q)
        self.sigma = np.load(FILEPATH +'models/sigma.npy')
        self.mu = np.load(FILEPATH + 'models/prior.npy')
        tmp = np.load(FILEPATH + 'models/grid.npy')
        self.lats = tmp[:,2]
        self.lons = tmp[:,3]
        self.x = tmp[:,0]
        self.y = tmp[:,1]

    def sample(self,n = 1):
        """Samples the GMRF. Only used to test.

        Args:
            n (int, optional): Number of realizations. Defaults to 1.
        """
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*self.sigma[1]
        return(data)

    def cholesky(self,Q):
        """A function calculating the cholesky decoposition of a positive definite precision matrix of the GMRF. Uses the c++ package Cholmod

        Args:
            Q ([N,N] sparse csc matrix): Sparse matrix from scipy.sparse
        """
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def candidate(self,ks,n=40):
        """Returns the marginal variance of all location given that a location (ks) in the GMRF has been measured.
        Uses Monte Carlo samples to calculate the marginal variance for all locations.

        Args:
            ks (integer): Index of the location been measured in the GRMF.
            n (int, optional): Number of samples used in the Monte Carlo estimate. Defaults to 40.
        """
        Q = self.Q.copy()
        Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2 
        Q_fac = self.Q_fac
        Q_fac.cholesky_inplace(Q)
        return(self.mvar(Q_fac = Q_fac,n=n))

    def update(self, rel, ks):
        """Update mean and precision of the GMRF given some measurements in the field.

        Args:
            rel ([k,1]-array): k number of measurements of the GMRF. (k>0).
            ks ([k,]-array): k number of indicies describing the index of the measurment in the field. 
        """
        mu = self.mu.reshape(-1,1)
        S = self.Stot[ks,:]
        self.Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2
        self.Q_fac.cholesky_inplace(self.Q)
        mu = mu - self.Q_fac.solve_A(S.transpose()@(S@mu-rel)*1/self.sigma[0]**2)
        self.mu = mu.flatten()
        #self.Q[ks, ks] = self.Q[ks, ks] + 1 / self.sigma[0] ** 2
        #F = np.zeros(self.M * self.N * self.P)
        #F[ks] = 1
        #V = self.Q_fac.solve_A(F.transpose())
        #W = F @ V + self.sigma[0] ** 2 + self.sigma[1] ** 2
        #U = V / W
        #c = F @ self.mu - rel
        #self.mu = self.mu - U.transpose() * c
        #self.Q_fac = cholesky(self.Q)

    

    def mvar(self,Q_fac = None, n=40):
        """Monte Carlo Estimate of the marginal variance of a GMRF.

        Args:
            Q_fac (Cholmod object, optional): Cholmod cholesky object. Defaults to None.
            n (int, optional): Number of samples used in the Monte Varlo estimate. Defaults to 40.
        """
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        return(data.var(axis = 1))



