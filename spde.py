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

    
    def reduce(self,x,y,z):
        tx,ty,tz = np.meshgrid(np.arange(x[0],x[1]),np.arange(y[0],y[1]),np.arange(z[0],z[1]))
        tx = tx.flatten()
        ty = ty.flatten()
        tz = tz.flatten()
        ks = ty*self.M*self.P + tx*self.P + tz
        self.Q = self.Q[ks,:][:,ks]
        self.Q_fac = cholesky(self.Q)
        self.M = x[1]-x[0]
        self.N = y[1]-y[0]
        self.P = z[1]-z[0]
        self.n = self.M*self.N*self.P

    # fix par for models
    def define(self, model = 3):
        if (model==1):
            tmp = np.load('./models/SINMOD-SA.npy')
        elif (model==2):
            tmp = np.load('./models/SINMOD-NI.npy')
        elif (model==3):
            tmp = np.load('./models/SINMOD-NA2-full.npy')
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(20250,20250))
        self.Q_fac = cholesky(self.Q)
        self.sigma = np.load('./models/sigma.npy')
        self.mu = np.load('./models/prior.npy')
        tmp = np.load('./models/grid.npy')
        self.lats = tmp[:,2]
        self.lons = tmp[:,3]
        self.x = tmp[:,0]
        self.y = tmp[:,1]

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

    def candidate(self,pos,n=40):
        ks = pos[1]*self.M*self.P + pos[0]*self.P + pos[2]
        Q = self.Q.copy()
        Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2 
        Q_fac = cholesky(Q)
        return(self.mvar(Q_fac = Q_fac,n=n))

    def update(self,rel,pos):
        ks = pos[1]*self.M*self.P + pos[0]*self.P + pos[2]
        self.Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2 
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



