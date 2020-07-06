import numpy as np
import torch
from torch import nn
from scipy import sparse
import pdb
from scipy.stats import poisson
from scipy.stats import binom

householdprobs = [0.2837, 0.3451, 0.1507, 0.1276, 0.0578, 0.0226, 0.0125]
chiprobs = [0.475, 0.2, 0.125, 0.1, 0.07, 0.03]

class Env:
    def __init__(self, Ninf, Ntotal, alpha = 0.03, meancontacts=2):
        Ntotal = int(Ntotal)
        self.alpha = alpha
        self.Ninf = Ninf
        self.Ntotal = Ntotal
        self.Cstatic = self.genChome()
        
        self.homesize = np.array(self.Cstatic.sum(axis = 0))[0].astype('int16') + 1
        
        self.I = np.zeros(Ntotal, dtype = 'bool')
        perm = np.random.permutation(Ntotal)
        self.I[perm[0:Ninf]] = True
        self.meancontacts = meancontacts
        self.rho = poisson.rvs(meancontacts, size=Ntotal)
        
        self.kappa0 = np.maximum(np.random.normal(1, 0.25, Ntotal), 0)
        sigmamu = np.random.multivariate_normal([1, 1], [[0.25, 0], [0, 0.25]], size = Ntotal)
        self.sigma0 = np.maximum(sigmamu[:, 0], 0)
        self.mu0 = np.minimum(np.maximum(sigmamu[:, 1], 0), 4)        
        
        self.tau0 = np.maximum(poisson.rvs(14, size=Ntotal), 8)
        self.chi = binom.rvs(5, 0.05, size=Ntotal)
        self.chi[self.I] = binom.rvs(5, self.mu0[self.I] / 4)
        
        self.S = ~self.I
        self.R = np.zeros(Ntotal, dtype = 'bool')
        self.D = np.zeros(Ntotal, dtype = 'bool')
        
        self.Idays = np.zeros(Ntotal) # number of days since infection
        self.Idays[perm[0:Ninf]] = 1
        self.Vdays = np.zeros(Ntotal) # number of days since vaccination
        
    def genChome(self):
        Ntotal = self.Ntotal
        sizes = np.random.choice(np.arange(1, 8), Ntotal, p = householdprobs)
        cumsumsizes = np.cumsum(sizes)
        nz = np.nonzero(cumsumsizes >= Ntotal)[0]
        Nhouseholds = nz[0] + 1
        if cumsumsizes[Nhouseholds - 1] > Ntotal:
            sizes[Nhouseholds - 1] = Ntotal - cumsumsizes[Nhouseholds - 2]
        cumsumsizes = np.cumsum(sizes)
        self.homesizes = sizes[0:Nhouseholds]
        self.homestarts = cumsumsizes[0:Nhouseholds] - sizes[0:Nhouseholds]
        x = [np.array(np.nonzero(1 - np.eye(sizes[i]))) + cumsumsizes[i] - sizes[i] for i in range(Nhouseholds) if sizes[i] > 1]
        x = np.concatenate(x, axis = 1)
        return sparse.coo_matrix((np.ones(x.shape[1]), (x[0], x[1])), shape=(Ntotal, Ntotal))
    
    def genCrandom(self):
        N = self.Ntotal
        nc = poisson.rvs(self.rho / 2)
        i1 = np.repeat(np.arange(N), nc)
        i2 = i1[np.random.permutation(i1.size)]
        v = np.ones(i1.size)
        c = sparse.coo_matrix((v, (i1, i2)), shape=(N, N))
        return c
    
    def step1(self, useCrandom=True):
        numD0 = np.sum(self.D)
        if useCrandom:
            Crandom = self.genCrandom()  #generate random contacts for the day
        else:
            Crandom = None
        dI = self.delI(Crandom)  #
        
        self.Idays[self.Idays > 0] += 1
        self.Idays[dI] = 1
        self.chi[dI] = binom.rvs(5, self.mu0[dI] / 4)
        
        dR = self.delR()
        Iprime = self.I + dI
        Iprime = Iprime ^ dR
        
        self.R = self.R + dR
        self.S = self.S*(~Iprime)
        
        dD = self.delD(Iprime)
        self.I = Iprime*(~dD)
        self.D = self.D + dD
        
        numD1 = np.sum(self.D)
        
        self.Vdays[self.Vdays > 0] += 1
        return np.sum(dI)
    
    def step2(self):
        vaccsuccess = (self.Vdays == self.tau0)*self.S
        self.S = self.S*(~vaccsuccess)
        self.R = self.R + vaccsuccess
        return (np.sum(self.S), np.sum(self.I), np.sum(self.R), np.sum(self.D))
    
    def vaccinate(self, individuals):
        i1 = np.sum(self.Vdays > 0)
        self.Vdays[individuals] = np.maximum(self.Vdays[individuals], 1*(~self.D[individuals]))
        i2 = np.sum(self.Vdays > 0)
        return (i2 - i1)
    
    def getsigma(self):
        return self.sigma0*(np.minimum(1 - (self.Vdays - 7)/(self.tau0 - 7), 1))
    
    def getkappa(self):
        return self.kappa0
    
    def getgamma(self):
        return np.minimum(self.Idays, 21)*(1 / (1 + 0.2*self.chi*self.I))
    
    def delI(self, crandom):
        prI = self.alpha*(self.getsigma()*self.S)*(self.Cstatic@(self.getkappa()*self.I))
        if crandom != None:
            prI += self.alpha*(self.getsigma()*self.S)*(crandom@(self.getkappa()*self.I))
            prI += self.alpha*(self.getsigma()*self.S)*(crandom.T@(self.getkappa()*self.I))
        u = np.random.random(self.Ntotal)
        return (prI > u)
    
    def delD(self, iprime):
        prD = (np.exp(2*self.chi) - 1)
        u = np.random.random(self.Ntotal)
        prD *= iprime
        prD *= 0.000003
        return (prD > u)
    
    def delR(self):
        prR = self.getgamma()*self.I
        prR *= 0.003
        u = np.random.random(self.Ntotal)
        return (prR > u)

def getdata(env_in, stdscale=[1.41454338, 0.4903974 , 0.48974935, 0.2501072 , 1.06461081,
       1.56293792, 1.42078538, 0.43015375, 0.42896917, 0.21571087,
       1.20871716], muref=[1.999298, 1.00434202, 1.00460432, 1.00001219, 0.921485,
       3.236878, 3.135669, 1.39322175, 1.3930576, 1.19671717,
       1.748563], appendone = False, addD=False, addV=False):
    N = env_in.Ntotal
    numdim = 11
    if stdscale == 1:
        stdscale = 1
    muref = 0
    out = np.zeros((N, numdim))
    out[:, 0] = env_in.rho
    out[:, 1] = env_in.sigma0
    out[:, 2] = env_in.mu0
    out[:, 3] = env_in.kappa0
    out[:, 4] = env_in.chi
    out[:, 5] = env_in.homesize
    
    out[:, 6] = np.repeat(np.maximum.reduceat(env_in.rho, env_in.homestarts), env_in.homesizes) # max rho in family
    out[:, 7] = np.repeat(np.maximum.reduceat(env_in.sigma0, env_in.homestarts), env_in.homesizes)
    out[:, 8] = np.repeat(np.maximum.reduceat(env_in.mu0, env_in.homestarts), env_in.homesizes)
    out[:, 9] = np.repeat(np.maximum.reduceat(env_in.kappa0, env_in.homestarts), env_in.homesizes)
    out[:, 10] = np.repeat(np.maximum.reduceat(env_in.chi, env_in.homestarts), env_in.homesizes)
    outscaled = (out - muref) / stdscale
    if appendone:
        outscaled = np.vstack((np.ones((outscaled.shape[0], 1)), outscaled))
    if addD:
        outscaled = np.hstack((outscaled, env_in.D.reshape((-1, 1))))
    if addV:
        outscaled = np.hstack((outscaled, (env_in.Vdays > 0).reshape((-1, 1))))
    return outscaled
        
def trialEnv(policy = None, Ninf=10, N=2000, T=125, lam0 = 10, usevac=True, appendone=False, usepredict=False, 
            addD=False, addV=False):
    if policy is None:
        userandom = True
    else:
        userandom = False
    usedot = False
    if type(policy) == np.ndarray:
        usedot = True
    localenv = Env(Ninf, N)
    
    SIRD = np.zeros((T, 4))
    for t in range(T):
        nonv = np.nonzero((localenv.Vdays == 0)*(localenv.D == False))[0]
        lam = np.minimum(nonv.size, lam0)
        data = getdata(localenv, appendone=appendone, addD=addD, addV=addV)
        localenv.step1()
        if usevac:
            if userandom:
                vac = np.random.choice(nonv, lam, replace=False)
            elif usedot:
                scores = data@policy
                toplam = np.argpartition(scores[nonv], -lam)[-lam:] #greedy, i.e. those with largest 
                vac = nonv[toplam]
            elif usepredict:
                scores = policy.predict(data)
                toplam = np.argpartition(scores[nonv], -lam)[-lam:] #greedy, i.e. those with largest 
                vac = nonv[toplam]
            else:
                with torch.no_grad():
                    torchdata = torch.FloatTensor(data)
                    scores = policy(torchdata).numpy().flatten()
                toplam = np.argpartition(scores[nonv], -lam)[-lam:] #greedy, i.e. those with largest score
                vac = nonv[toplam]
            localenv.vaccinate(vac)
        s, i, r, d = localenv.step2()
        SIRD[t] = [s, i, r, d]
    return SIRD[-1, 3]