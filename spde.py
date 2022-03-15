import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky

class spde:
    def __init__(self, model = 6):
        self.define(model = model)
        self.M = 45
        self.N = 45
        self.P = 10
        self.n = self.M*self.N*self.P

    # fix par for models
    def define(self, model = 3):
        if (model==1):
            tmp = np.load('./models/SINMOD-SA.npy')
        elif (model==2):
            tmp = np.load('./models/SINMOD-NI.npy')
        elif (model==3):
            tmp = np.load('./models/SINMOD-NA.npy')
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(20250,20250))
        self.Q_fac = cholesky(self.Q)
        self.sigma = np.load('./models/sigma.npy')
        self.mu = np.load('./models/prior.npy')

    def sample(self,n = 1):
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*self.sigma[1]
        return(data)

    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def candidate(self,pos):
        ks = pos[1]*self.M*self.P + pos[0]*self.P + pos[2]
        Q = self.Q.copy()
        Q[ks,ks] = self.Q[ks,ks] + self.sigma[0]**2 + self.sigma[1]**2
        Q_fac = cholesky(Q)
        return(self.mvar(Q_fac = Q_fac))

    def update(self,rel,pos):
        ks = pos[1]*self.M*self.P + pos[0]*self.P + pos[2]
        self.Q[ks,ks] = self.Q[ks,ks] + self.sigma[0]**2 + self.sigma[1]**2
        F = np.zeros(self.M*self.N*self.P)
        F[ks] = 1
        V = self.Q_fac.solve_A(F.transpose())
        W = F@V + self.sigma[0]**2 + self.sigma[1]**2
        U = V/W
        c = F@self.mu - rel
        self.mu = self.mu - U.transpose()*c
        self.Q_fac = cholesky(self.Q)

    def mvar(self,Q_fac = None, n=40):
        if Q_fac is None: 
            Q_fac = self.Q_fac
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        return(data.var(axis = 1))



