import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky
from Config.Config import FILEPATH


class spde:
    def __init__(self, model = 2, reduce = False, method = 1):
        """Initialize model

        Args:
            model (int, optional): Doesn't do anything. Defaults to 2.
            reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
            method (int, optional): If model should contain fixed effects on the SINMOD mean. Defaults to 1.
        """
        # grid
        self.M = 45
        self.N = 45
        self.P = 11
        self.n = self.M*self.N*self.P
        
        # define model from files
        tmp = np.load(FILEPATH + 'models/SINMOD-NAf.npy') # load fitted precision matrix Q
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n)) 
        self.Q_fac = cholesky(self.Q)  # calculate cholesky decomoposition
        self.sigma = np.load(FILEPATH + 'models/sigma.npy') # measurement noise robot [0] and SINMOD fitted noise [1]
        self.mu = np.load(FILEPATH + 'models/prior.npy') # salinity prior SINMOD
        tmp = np.load(FILEPATH + 'models/grid.npy') # loading grid data
        self.lats = tmp[:,2] # min & max latitudes
        self.lons = tmp[:,3] # min & max longitudes
        self.x = tmp[:,0] # min & max x grid location
        self.y = tmp[:,1] # min & max y grid locations
        
        self.reduced = reduce # using a reduced grid
        self.method = method # method 2 is with fixed effects on the SINMOD mean
        if self.reduced:
            self.reduce() 
        self.Stot = sparse.eye(self.n).tocsc()
        self.mu3 = self.mu
        if self.method == 2: 
            # reshaping Q for fixed effects
            self.Q.resize((self.n+2,self.n+2))
            self.Q = self.Q.tolil()
            self.Q[self.n,self.n] = 0.01
            self.Q[self.n+1,self.n+1] = 0.1
            self.Q = self.Q.tocsc()
            self.Q_fac = cholesky(self.Q)
            # setting mean
            self.mu2 = np.hstack([np.zeros(self.n),0,1]).reshape(-1,1) # Mean of random effect and betas
            self.mu3 = self.mu
            self.beta0 = 0
            self.beta1 = 1

            self.Stot.resize((self.n,self.n+2))
            self.Stot = self.Stot.tolil()
            self.Stot[:,self.n] = np.ones(self.n)
            self.Stot[:,self.n+1] = self.mu3
            self.Stot = self.Stot.tocsc()

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

    def sample(self,n = 1):
        """Samples the GMRF. Only used to test.

        Args:
            n (int, optional): Number of realizations. Defaults to 1.
        """
        if self.method == 2:
            z = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n)
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
            data = data[:self.n,:] + self.mu3.reshape(-1,1) + np.random.normal(size = self.n*n).reshape(self.n,n)*self.sigma[1]
        else:
            z = np.random.normal(size = self.n*n).reshape(self.n,n)
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*self.sigma[1] + self.mu3
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

    def save(self):
        """Saves conditional mean and marginal variance
        """
        np.save(FILEPATH + 'models/mucond.npy', self.mu)
        np.save(FILEPATH + 'models/smvarcond.npy', self.mvar())

    def candidate(self,ks,n=150):
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
        if ks.size>0:
            if self.method == 2:
                S = self.Stot[ks,:]
                self.Q = self.Q + S.transpose()@S*1/self.sigma[0]**2
                self.Q_fac.cholesky_inplace(self.Q)
                self.mu2 = self.mu2 - self.Q_fac.solve_A(S.transpose()@(S@self.mu2 - rel)*1/self.sigma[0]**2)
                self.mu = self.mu2[:self.n,0] + self.mu2[self.n,0] + self.mu3*self.mu2[self.n+1,0]
                self.beta0 = self.mu2[self.n]
                self.beta1 = self.mu2[self.n+1]
            else:
                mu = self.mu.reshape(-1,1)
                S = self.Stot[ks,:]
                self.Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2
                self.Q_fac.cholesky_inplace(self.Q)
                mu = mu - self.Q_fac.solve_A(S.transpose()@(S@mu-rel)*1/self.sigma[0]**2)
                self.mu = mu.flatten()

    def mvar(self,Q_fac = None, n=150):
        """Monte Carlo Estimate of the marginal variance of a GMRF.

        Args:
            Q_fac (Cholmod object, optional): Cholmod cholesky object. Defaults to None.
            n (int, optional): Number of samples used in the Monte Varlo estimate. Defaults to 40.
        """
        if Q_fac is None:
            Q_fac = self.Q_fac
        if self.method == 2:
            z = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n)
            data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
            data = data[:self.n,:] + data[self.n,:] + self.mu3.reshape(-1,1)*data[self.n+1,:].reshape(1,-1)
        else:
            z = np.random.normal(size = self.n*n).reshape(self.n,n)
            data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        return(data.var(axis = 1))

    def resetQ(self):
        """Resets Q to initial values for the same grid
        """
        self.M = 45
        self.N = 45
        self.P = 11
        self.n = self.M*self.N*self.P
        tmp = np.load(FILEPATH + 'models/SINMOD-NAf.npy') # load fitted precision matrix Q
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
        self.Q_fac = cholesky(self.Q)
        if self.reduced:
            self.reduce()
        self.Stot = sparse.eye(self.n).tocsc()
        if self.method == 2:
            self.Q.resize((self.n + 2, self.n + 2))
            self.Q = self.Q.tolil()
            self.Q[self.n, self.n] = 0.01
            self.Q[self.n + 1, self.n + 1] = 0.1
            self.Q = self.Q.tocsc()
            self.Q_fac = cholesky(self.Q)
            
            self.Stot.resize((self.n,self.n+2))
            self.Stot = self.Stot.tolil()
            self.Stot[:,self.n] = np.ones(self.n)
            self.Stot[:,self.n+1] = self.mu3
            self.Stot = self.Stot.tocsc()
            self.mu = self.mu2[:self.n,0] + self.mu2[self.n,0] + self.mu3*self.mu2[self.n+1,0]
        print("Q: ", self.Q.shape)


            