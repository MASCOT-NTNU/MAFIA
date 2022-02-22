# 1. get prior Q
# 2. sample 
# 3. Update from samples
# 4. 

import sys, getopt
import numpy as np
import os
from grid import Grid
from scipy import sparse
from sksparse.cholmod import cholesky
from ah3d2 import AH

class spde:
    def __init__(self, model = 3):
        self.grid = Grid()
        self.define(model = model)

    # fix par for models
    def define(self, model = 3):
        if (self.model==1):
            self.StatIso()
        elif (self.model==2):
            self.StatAnIso()
        elif (self.model==3):
            self.NonStatAnIso()

    def StatIso(self):
        V = self.grid.hx*self.grid.hy*self.grid.hz
        n = self.grid.M*self.grid.N*self.grid.P
        Dv = V*sparse.eye(self.n)
        par = np.load('SINMOD-SI.npz')['par']*1
        kappa = par[0]
        gamma = par[1]
        tau = par[2]
        self.sigma = np.sqrt(1/np.exp(tau))
        Hs = np.diag(np.exp([gamma,gamma,gamma]))+ np.zeros((n,6,3,3))
        Dk =  sparse.diags([np.exp(kappa)]*n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(n,n))
        A_mat = Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)

    def StatAnIso(self):
        V = self.grid.hx*self.grid.hy*self.grid.hz
        n = self.grid.M*self.grid.N*self.grid.P
        Dv = V*sparse.eye(self.n)
        par = np.load('SINMOD-SA.npz')['par']*1
        kappa = par[0]
        gamma = par[1]
        vx = par[2]
        vy = par[3]
        vz = par[4]
        rho1 = par[5]
        rho2 = par[6]
        tau = par[7]
        self.sigma = np.sqrt(1/np.exp(tau))
        vv = np.array([vx,vy,vz])
        vb1 = np.array([-vy,vx,vz])
        vb2 = np.array([vy*vz - vz*vx,-vz*vy - vx*vz,vx**2 + vy**2])
        ww = rho1*vb1 + rho2*vb2
        Hs = np.diag(np.exp([gamma,gamma,gamma])) + vv[:,np.newaxis]*vv[np.newaxis,:] + ww[:,np.newaxis]*ww[np.newaxis,:] + np.zeros((n,6,3,3))
        Dk =  sparse.diags([np.exp(kappa)]*n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(n,n))
        A_mat = Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
    
    def NonStatAnIso(self):
        V = self.grid.hx*self.grid.hy*self.grid.hz
        n = self.grid.M*self.grid.N*self.grid.P
        Dv = V*sparse.eye(self.n)
        par = np.load('SINMOD-SA.npz')['par']*1
        self.grid.basisH()
        self.grid.basisN()
        kappa = par[0]
        gamma = par[1]
        vx = par[2]
        vy = par[3]
        vz = par[4]
        rho1 = par[5]
        rho2 = par[6]
        tau = par[7]
        self.sigma = np.sqrt(1/np.exp(tau))
        pg = np.exp(self.grid.evalBH(par = gamma))
        vv = np.stack([self.grid.evalBH(par = vx),self.grid.evalBH(par = vy),self.grid.evalBH(par = vz)],axis=2)
        vb1 = np.stack([-vv[:,:,1],vv[:,:,0],vv[:,:,2]],axis=2)
        vb2 = np.stack([vv[:,:,1]*vv[:,:,2]-vv[:,:,2]*vv[:,:,0],-vv[:,:,2]*vv[:,:,1]-vv[:,:,0]*vv[:,:,2],vv[:,:,0]**2 + vv[:,:,1]**2],axis=2)
        ww = vb1*self.grid.evalBH(par = rho1)[:,:,np.newaxis] + vb2*self.grid.evalBH(par = rho2)[:,:,np.newaxis]
        Hs = (np.eye(3)*(np.stack([pg,pg,pg],axis=2))[:,:,:,np.newaxis]) + vv[:,:,:,np.newaxis]*vv[:,:,np.newaxis,:] + ww[:,:,:,np.newaxis]*ww[:,:,np.newaxis,:]
        
        Dk =  sparse.diags([np.exp(self.grid.evalB(kappa))]) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(n,n))
        A_mat = Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
        self.grid.dropMemory()

    def sample(self,n = 1):
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*self.sigma
        return(data)

    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def predict(self):
        return

    def update(self,sample,pos):
        return

    def mvar(self):
        return

    def cov(self):
        return




