#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import progressbar
from scipy.optimize import fsolve
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu

b=6
pho=99./100.

def gen_data(T,l,collection_input):
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    #print(l)
    tau=1./2**(l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    C=tau*H
    V=(R2@R2)*tau

    v=np.zeros((J+1,dim,1))
    z=np.zeros((J+1,dim_o,1))
    v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    z[0]=np.zeros((dim_o,1))


    for j in range(J):
        ## truth
        #M: how do we know this is the truth? 
        v[j+1] = L@v[j] + np.random.multivariate_normal(np.zeros(dim),W,(1)).T
        ## observation
        z[j+1] = z[j] + C@v[j+1] + np.random.multivariate_normal(np.zeros(dim_o),V,(1)).T
        
    return([z,v])
def gen_model(dim,amm,bmm,cmm):
    ## dim is dimension value
    dim_o=dim
    A1 = -rand(dim,dim,density=0.75).toarray()/1
    A2 = triu(A1, k=1).toarray()
    A = (diags(np.random.normal(-0.5,0,dim),0).toarray()/50 + A2 - A2.T)/cmm
    ## we denote R1^{1/2},R2^{1/2} as R1,R2 repectively for convenience
    ## Non-Identity covariance matrix
    R1=(identity(dim).toarray() + np.tri(dim,dim,1) - np.tri(dim,dim,-2))/amm
    R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/bmm
    #R2=identity(dim_o).toarray()/bmm
    H=rand(dim_o,dim,density=0.75).toarray()/20.
    m0=np.zeros(dim)+b
    C0=identity(dim).toarray()
    ## Collection of input 
    collection_input = [dim,dim_o,A,R1,R2,H,m0,C0]
    return collection_input


def KBF(T,l,lmax,z,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    ## C: dim_o*dim matrix
    C=tau*H
    V=(R2@R2)*tau
    
    z=cut(T,lmax,l,z)
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    m[0]=b*np.ones((dim,1))
    c[0]=C0
    
    for j in range(J):
        ## prediction mean-dim*1 vector
        mhat=L@m[j]
        ## prediction covariance-dim*dim matrix
        chat=L@c[j]@(L.T)+W
        ## innovation-dim_o*1 vector
        d=(z[j+1]-z[j])-C@mhat
        ## Kalman gain-dim*dim_o vector
        K=(chat@(C.T))@la.inv(C@chat@(C.T)+V)
        ## update mean-dim*1 vector
        m[j+1]=mhat+K@d
        ## update covariance-dim*dim matrix
        c[j+1]=(I-K@C)@chat
        
    return([m,c])


# EnKBF and Coupled EnKBF


def EnKBF(T,l,lmax,z,N,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    z=cut(T,lmax,l,z)
    
    ## This gives a dim*N matrix
    x = np.random.multivariate_normal(m0,C0,N).T
    ## A dim*1 vector
    m[0]=(np.mean(x, axis=1)).reshape(dim,1)
    ## dim*dim matrix
    c[0]=((x-m[0])@((x-m[0]).T)) /(N-1)
    inv=la.inv(R2)@la.inv(R2)
    for j in range(J):
        dw = np.random.multivariate_normal(np.zeros(dim),dt*I,N).T
        dv = np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,N).T
        ## A@x:dim*N R1@dw:dim*N c[j]@(H.T):dim*dim_o z[j+1]-z[j]:dim_o*1 H@x*dt:dim_o*N R2*dv:dim_o*N 
        ## x-m[j]:dim*N c[j]:dim*dim
        
        step1=(((x-m[j]).T)@(H.T))
        step2=step1@(inv)
        step3=step2@( (z[j+1]-z[j]) - (H@x*dt + R2@dv) )
        step4=(x-m[j])@step3 /(N-1)
        
        x = x + A@x*dt + R1@dw + step4
        m[j+1] = (np.mean(x, axis=1)).reshape(dim,1)

    return([m,c])





def CEnKBF(T,l,lmax,z,N,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**(l-1))
    I=identity(dim).toarray()
    I1=identity(dim_o).toarray()
    dt=2**(-l)
    dt1=2**(-l+1)
    
    m=np.zeros((J*2+1,dim,1))
    m1=np.zeros((J+1,dim,1))
    c=np.zeros((J*2+1,dim,dim))
    c1=np.zeros((J+1,dim,dim))
    z1=cut(T,lmax,l-1,z)
    z=cut(T,lmax,l,z)
    
    ## This gives a dim*N matrix
    x = np.random.multivariate_normal(m0,C0,N).T
    x1 = x
    ## A dim*1 vector
    m[0]=(np.mean(x, axis=1)).reshape(dim,1)
    m1[0]=m[0]
    ## dim*dim matrix
    c[0]=((x-m[0])@((x-m[0]).T)) /(N-1)
    c1[0]=c[0]
    
    dw=np.zeros((2,dim,N))
    dv=np.zeros((2,dim_o,N))
    inv=la.inv(R2)@la.inv(R2)
    for j in range(J):
        for s in range(2):
            dw[s] = np.random.multivariate_normal(np.zeros(dim),dt*I,N).T
            dv[s] = np.random.multivariate_normal(np.zeros(dim_o),dt*I1,N).T
            ## A@x:dim*N R1@dw:dim*N c[j]@(H.T):dim*dim_o z[j+1]-z[j]:dim_o*1 H@x*dt:dim_o*N R2*dv:dim_o*N 
            ## x-m[j]:dim*N c[j]:dim*dim

            step1=(((x-m[2*j+s]).T)@(H.T))
            step2=step1@(inv)
            step3=step2@( (z[2*j+s+1]-z[2*j+s]) - (H@x*dt + R2@dv[s]) )
            step4=(x-m[2*j+s])@step3 /(N-1)

            x = x + A@x*dt + R1@dw[s] + step4
            m[2*j+s+1] = (np.mean(x, axis=1)).reshape(dim,1)
        
        step1=(((x1-m1[j]).T)@(H.T))
        step2=step1@(inv)
        step3=step2@( (z1[j+1]-z1[j]) - (H@x1*dt1 + R2@(dv[0]+dv[1])) )
        step4=(x1-m1[j])@step3 /(N-1)
        
        x1 = x1 + A@x1*dt1 + R1@(dw[0]+dw[1]) + step4
        m1[j+1] = (np.mean(x1, axis=1)).reshape(dim,1)
        c[j+1]=(x1-m1[j])@((x1-m1[j]).T)
    return([m,m1])



def DCEnKBF(T,l,lmax,z,N,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**(l-1))
    I=identity(dim).toarray()
    I1=identity(dim_o).toarray()
    dt=2**(-l)
    dt1=2**(-l+1)
    
    m=np.zeros((J*2+1,dim,1))
    m1=np.zeros((J+1,dim,1))
    c=np.zeros((J*2+1,dim,dim))
    c1=np.zeros((J+1,dim,dim))
    z1=cut(T,lmax,l-1,z)
    z=cut(T,lmax,l,z)
    
    ## This gives a dim*N matrix
    x = np.random.multivariate_normal(m0,C0,N).T
    x1 = x
    ## A dim*1 vector
    m[0]=(np.mean(x, axis=1)).reshape(dim,1)
    m1[0]=m[0]
    ## dim*dim matrix
    c[0]=((x-m[0])@((x-m[0]).T)) /(N-1)
    c1[0]=c[0] 
    inv=la.inv(R2)@la.inv(R2)
    dw=np.zeros((2,dim,N))
    dv=np.zeros((2,dim_o,N))
    for j in range(J):
        for s in range(2):
            dw[s] = np.random.multivariate_normal(np.zeros(dim),dt*I,N).T
            dv[s] = np.random.multivariate_normal(np.zeros(dim_o),dt*I1,N).T
            ## A@x:dim*N R1@dw:dim*N c[j]@(H.T):dim*dim_o z[j+1]-z[j]:dim_o*1 H@x*dt:dim_o*N R2*dv:dim_o*N 
            ## x-m[j]:dim*N c[j]:dim*dim

            step1=(((x-m[2*j+s]).T)@(H.T))
            step2=step1@(inv)
            step3=step2@( (z[2*j+s+1]-z[2*j+s]) - (H@(x+m[2*j+s])*dt)/2 )
            step4=(x-m[2*j+s])@step3 /(N-1)

            x = x + A@x*dt + R1@dw[s] + step4
            m[2*j+s+1] = (np.mean(x, axis=1)).reshape(dim,1)
        
        step1=(((x1-m1[j]).T)@(H.T))
        step2=step1@(inv)
        
        step3=step2@( (z1[j+1]-z1[j]) - (H@(x1+m1[j])*dt1)/2 )
        step4=(x1-m1[j])@step3 /(N-1)
        
        x1 = x1 + A@x1*dt1 + R1@(dw[0]+dw[1]) + step4
        m1[j+1] = (np.mean(x1, axis=1)).reshape(dim,1)
        
    return([m,m1])





def DEnKBF(T,l,lmax,z,N,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    z=cut(T,lmax,l,z)
    
    ## This gives a dim*N matrix
    x = np.random.multivariate_normal(m0,C0,N).T
    ## A dim*1 vector
    m[0]=(np.mean(x, axis=1)).reshape(dim,1)
    ## dim*dim matrix
    c[0]=((x-m[0])@((x-m[0]).T)) /(N-1)
    inv=la.inv(R2)@la.inv(R2)
    for j in range(J):
        dw = np.random.multivariate_normal(np.zeros(dim),dt*I,N).T
        dv = np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,N).T
        ## A@x:dim*N R1@dw:dim*N c[j]@(H.T):dim*dim_o z[j+1]-z[j]:dim_o*1 H@x*dt:dim_o*N R2*dv:dim_o*N 
        ## x-m[j]:dim*N c[j]:dim*dim
        
        step1=(((x-m[j]).T)@(H.T))
        step2=step1@(inv)
        # Only the "innovation" term here changes to the "deterministic" version
        step3=step2@( (z[j+1]-z[j]) - (H@(x+m[j])*dt)/2 )
        step4=(x-m[j])@step3 /(N-1)
        
        x = x + A@x*dt + R1@dw + step4
        m[j+1] = (np.mean(x, axis=1)).reshape(dim,1)

    return([m,c])

def DCEnKBF(T,l,lmax,z,N,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**(l-1))
    I=identity(dim).toarray()
    I1=identity(dim_o).toarray()
    dt=2**(-l)
    dt1=2**(-l+1)
    
    m=np.zeros((J*2+1,dim,1))
    m1=np.zeros((J+1,dim,1))
    c=np.zeros((J*2+1,dim,dim))
    c1=np.zeros((J+1,dim,dim))
    z1=cut(T,lmax,l-1,z)
    z=cut(T,lmax,l,z)
    
    ## This gives a dim*N matrix
    x = np.random.multivariate_normal(m0,C0,N).T
    x1 = x
    ## A dim*1 vector
    m[0]=(np.mean(x, axis=1)).reshape(dim,1)
    m1[0]=m[0]
    ## dim*dim matrix
    c[0]=((x-m[0])@((x-m[0]).T)) /(N-1)
    c1[0]=c[0]
    inv=la.inv(R2)@la.inv(R2)
    dw=np.zeros((2,dim,N))
    dv=np.zeros((2,dim_o,N))
    for j in range(J):
        for s in range(2):
            dw[s] = np.random.multivariate_normal(np.zeros(dim),dt*I,N).T
            dv[s] = np.random.multivariate_normal(np.zeros(dim_o),dt*I1,N).T
            ## A@x:dim*N R1@dw:dim*N c[j]@(H.T):dim*dim_o z[j+1]-z[j]:dim_o*1 H@x*dt:dim_o*N R2*dv:dim_o*N 
            ## x-m[j]:dim*N c[j]:dim*dim

            step1=(((x-m[2*j+s]).T)@(H.T))
            step2=step1@(inv)
            step3=step2@( (z[2*j+s+1]-z[2*j+s]) - (H@(x+m[2*j+s])*dt)/2 )
            step4=(x-m[2*j+s])@step3 /(N-1)

            x = x + A@x*dt + R1@dw[s] + step4
            m[2*j+s+1] = (np.mean(x, axis=1)).reshape(dim,1)
        
        step1=(((x1-m1[j]).T)@(H.T))
        step2=step1@(inv)
        
        step3=step2@( (z1[j+1]-z1[j]) - (H@(x1+m1[j])*dt1)/2 )
        step4=(x1-m1[j])@step3 /(N-1)
        
        x1 = x1 + A@x1*dt1 + R1@(dw[0]+dw[1]) + step4
        m1[j+1] = (np.mean(x1, axis=1)).reshape(dim,1)
        
    return([m,m1])




# Copies the i*2^(lmax-l) positions for i=0,1,...,T*2^l
def cut(T,lmax,l,v):
    ind = np.arange(T*2**l+1)
    rtau = 2**(lmax-l)
    w = v[ind*rtau]
    return(w)


# Testing Floor: KBF

# Ensemble Kalman Bucy Filter Function ： Euler

# #### Outer shell of the Unbiased estimators

# Sequence of samples


#gives a sequence of numbers [N0,N1,...,Np]
def nseq(p,N0):
    #N0, The base number of particles has to be considerably big(N0>2) to avoid overflow.
    seq=np.concatenate(([0],N0*np.array([2**i for i in range(p+1)])))
    seq_diff=seq[1:]-seq[:-1]
    return [seq,seq_diff]


# The following function computes the (3.2) in the paper, $\Xi_{l,p}\mathbb{P}_P(p)$.

# Modified version:
# Arguments of this function 
# 
# p:  level of number of samples 
# 
# N0:  base number of samples 
# 
# l:  level of time discretization, $l$ $\in$ ${0,1,...,lmax=lz-l0}$ 
# 
# l0: shift of the levels, i.e. the actual time step for a value of l is $2^{-l0-l}$
# 
# lz:  level of discretization of the observations 
# 
# phi: function applied to the EnKBF "particles"
# 
# T: time span of the filter
# 
# z: observations

def plevels(p,N0,l,l0,lz,T,z,collection_input):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=EnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
        else:
            [m,c]=EnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            xi= (1./seq[-1])*funct_val    
    if l_original>0:
        if p==0:
            [m,m1]=CEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(m[-1].T)[0]
            xi1=funct_val
            funct_val2[:,0]=(m1[-1].T)[0]
            xi2=funct_val2
            xi=xi1-xi2
        
        if p>0:
            
            for i in range(p+1):
                [m,m1]=CEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]             
                
            xi1= ((1./seq[-1])-(1./seq[-2]))*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]
            xi2= (1./seq[-1]-1./seq[-2])*np.sum(funct_val2[:,:-1],axis=1)+funct_val2[:,-1]/seq[-1]
            xi=(xi1-xi2).reshape((dim,1))
    return xi

def phi2(x):
    return x



def plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    N=seq[-1]
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=EnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
            xi=xi/pdf_p[p]
        else:
            [m,c]=EnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #xi= (1./seq[-1])*funct_val    
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            #funct_val[:,0]=np.sum(phi(x),axis=1)
            xi= (1./seq[-1])*funct_val/pdf_p[p]    
    if l_original>0:
        
        if p==0:
            #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[p]))
            #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[p]))
            #w1=np.cumsum(dw1,axis=0) 
            #w2=np.cumsum(dw2,axis=0) 
            #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
            #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
            [m,m1]=CEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #funct_val[:,0]=np.sum(phi(x),axis=1).T
            xi1= (1./seq[-1])*funct_val
            #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val2[:,0]=(seq_diff[0]*m1[-1].T)[0]
            xi2= (1./seq[-1])*funct_val2
            xi=xi1-xi2
        
        if p>0:
            mult_constants=np.array([(1/seq[i+1]-1/seq[i])/(1-cdf_p[i-1]) for i in range(1,p+1)])
            vector_constants=np.zeros(p)
            vector_constants2=seq[1:]*np.concatenate(([1],1-cdf_p[:p]))
            for i in range(p):
                vector_constants[i]=np.sum(mult_constants[i:])
            for i in range(p+1):
                #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[i]))
                #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[i]))
                #w1=np.cumsum(dw1,axis=0) 
                #w2=np.cumsum(dw2,axis=0) 
                #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
                #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
                [m,m1]=CEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
                #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[i],collection_input,w1,w2)
                #funct_val2[:,i]=np.sum(phi(x),axis=1)   
            term1= np.sum(vector_constants*(funct_val[:,:-1]-funct_val2[:,:-1]),axis=1)
            term2=np.sum((funct_val-funct_val2)/vector_constants2,axis=1)
            xi=(term1+term2).reshape(dim,1)
                
            
    return xi

# Sampling function:
# 
# This functions takes as argument an array representing the cumulative distribution of a random variable,and samples from it. 
# 
# cumulative[0]=F(1)<cumulative[1]=F(2)<...<cumulative[n-1]=cumulative[-1]=F(n)=1.

# In[10]:


def sampling(cumulative):
    u=np.random.uniform()
    for i in range(len(cumulative)):
        if cumulative[i]>u:
            p_sampled=i
            break
        
    return p_sampled


# Cumulatives

# Creates a pdf(and its cdf) $P(x)$ $x\in \{0,1,2,...,lmax\}$
# 
# **Note:** To change the measure functions of the estimators it is necessary to change them directly into the functions single_term() and coupled_sum(). Don't forget that the functions have differents arguments. example: rh0.

# In[11]:


def DF_l(lmax):
    array=np.arange(lmax+1)
    pdf=2**(-1.5*array)
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return pdf, cdf



def DF_p(lmax,l):
    if (lmax-l)<=4:
        array=np.arange(lmax-l+1)
        pdf=2**(4-array)
        pdf=pdf/np.sum(pdf)
        cdf= np.cumsum(pdf)
    if lmax-l>=5:
        array1=np.arange(5)
        array2=np.arange(5,lmax-l+1)
        pdf1=2**(4-array1)
        pdf2=array2*(np.log2(array2))**2/(2**(array2))
        pdf=np.concatenate((pdf1,pdf2))
        pdf=pdf/np.sum(pdf)
        cdf= np.cumsum(pdf)
    return [pdf,cdf]


def DF_l1(lmax,rho):
    array=np.arange(lmax+1)
    pdf=2**(-1*array*rho)
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return pdf, cdf

def DF_p1(pmax):
    array=np.array(range(pmax+1))
    pdf=(array+1)*(np.log2(array+2))**2/(2**(array))
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return [pdf,cdf]


# Single term estimator 
# B: Number of i.i.d. samples of the unbiased estimator

# In[31]:


def single_term(B,N0,l0,lz,T,z,collection_input,pmax,lmax,rho):

    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    [pdf_l,cdf_l]=DF_l1(lmax,rho)
    #[pdf_l,cdf_l]=DF_l(lmax)
    for i in range(B):
        l=sampling(cdf_l)
        #[pdf_p,cdf_p]=DF_p1(pmax)
        #[pdf_p,cdf_p]=DF_p(lmax,l)
        [pdf_p,cdf_p]=DF_l1(pmax,rho)
        p=sampling(cdf_p)
        #print(i,l,p)
        print([p,l],i)
        xi=plevels(p,N0,l,l0,lz,T,z,collection_input)/(pdf_p[p]*pdf_l[l])
        estimators[i]=(xi.T)[0]
        pls[i]=[p,l]
    
    estimator_mean=np.mean(estimators,axis=0)
    #estimator_var=np.var(estimators,axis=0)

    return [estimator_mean,estimators,pls]



def coupled_sum(B,N0,l0,lz,T,z,collection_input,pmax,lmax):
    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    #[pdf_l,cdf_l]=DF_l(lmax)
    [pdf_l,cdf_l]=DF_l1(lmax,rho)
    for i in range(B):
        l=sampling(cdf_l)
        #[pdf_p,cdf_p]=DF_p1(pmax)
        #[pdf_p,cdf_p]=DF_p(lmax,l)
        [pdf_p,cdf_p]=DF_l1(pmax,rho)
        ##print(lmax,l,pdf_p)
        p=sampling(cdf_p)
        #print(i,l,p)
        xi=plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p)/pdf_l[l]
        #print(((xi.T)[0]))
        estimators[i]=(xi.T)[0]
        pls[i]=[p,l]
    
    estimators_mean=np.mean(estimators,axis=0)
    return [estimators_mean,estimators,pls]

# Primary **test** for the single term estimator: Good 

# Linear regression function:

# In[13]:


def coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    
    return np.asarray((b_0, b_1)) 


# Cost function depending on the levels

# In[14]:


def cost_function(T,pls,N0):
    cost=0
    for i in range(len(np.array(pls).T[0])):
        if pls[i,1]==0:
            cost+=2**(pls[i,0]+pls[i,1])
        else:
            cost+=2**(pls[i,0])*(2**(pls[i,1])+2**(pls[i,1]-1))
    cost=cost*T*N0
    return cost


# In this section we are going to compute the coupled sum estimators in the number of samples level.
# The only two changes with respect the function plevels is that we have the additional arguments cdf_p and pdf_p, and that we are compututing $\Xi_{l,p}$ instead of $\Xi_{l,p}\mathbb{P}_P(p)$

# In[15]:



def plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    N=seq[-1]
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=EnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
            xi=xi/pdf_p[p]
        else:
            [m,c]=EnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #xi= (1./seq[-1])*funct_val    
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            #funct_val[:,0]=np.sum(phi(x),axis=1)
            xi= (1./seq[-1])*funct_val/pdf_p[p]    
    if l_original>0:
        
        if p==0:
            #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[p]))
            #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[p]))
            #w1=np.cumsum(dw1,axis=0) 
            #w2=np.cumsum(dw2,axis=0) 
            #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
            #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
            [m,m1]=CEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #funct_val[:,0]=np.sum(phi(x),axis=1).T
            xi1= (1./seq[-1])*funct_val
            #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val2[:,0]=(seq_diff[0]*m1[-1].T)[0]
            xi2= (1./seq[-1])*funct_val2
            xi=xi1-xi2
        
        if p>0:
            mult_constants=np.array([(1/seq[i+1]-1/seq[i])/(1-cdf_p[i-1]) for i in range(1,p+1)])
            vector_constants=np.zeros(p)
            vector_constants2=seq[1:]*np.concatenate(([1],1-cdf_p[:p]))
            for i in range(p):
                vector_constants[i]=np.sum(mult_constants[i:])
            for i in range(p+1):
                #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[i]))
                #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[i]))
                #w1=np.cumsum(dw1,axis=0) 
                #w2=np.cumsum(dw2,axis=0) 
                #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
                #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
                [m,m1]=CEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
                #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[i],collection_input,w1,w2)
                #funct_val2[:,i]=np.sum(phi(x),axis=1)   
            term1= np.sum(vector_constants*(funct_val[:,:-1]-funct_val2[:,:-1]),axis=1)
            term2=np.sum((funct_val-funct_val2)/vector_constants2,axis=1)
            xi=(term1+term2).reshape(dim,1)
                
            
    return xi


# In[16]:


def coupled_sum(B,N0,l0,lz,T,z,collection_input,pmax,lmax):
    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    #[pdf_l,cdf_l]=DF_l(lmax)
    [pdf_l,cdf_l]=DF_l1(lmax,rho)
    for i in range(B):
        l=sampling(cdf_l)
        #[pdf_p,cdf_p]=DF_p1(pmax)
        #[pdf_p,cdf_p]=DF_p(lmax,l)
        [pdf_p,cdf_p]=DF_l1(pmax,rho)
        ##print(lmax,l,pdf_p)
        p=sampling(cdf_p)
        #print(i,l,p)
        xi=plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p)/pdf_l[l]
        #print(((xi.T)[0]))
        estimators[i]=(xi.T)[0]
        pls[i]=[p,l]
    
    estimators_mean=np.mean(estimators,axis=0)
    return [estimators_mean,estimators,pls]


# In[ ]:


# Copies the i*2^(lmax-l) positions for i=0,1,...,T*2^l
def cut(T,lmax,l,v):
    ind = np.arange(T*2**l+1)
    rtau = 2**(lmax-l)
    w = v[ind*rtau]
    return(w)

#Testing Floor: KBF

#Ensemble Kalman Bucy Filter Function ： Euler

#### Outer shell of the Unbiased estimators

#Sequence of samples

#gives a sequence of numbers [N0,N1,...,Np]
def nseq(p,N0):
    #N0, The base number of particles has to be considerably big(N0>2) to avoid overflow.
    seq=np.concatenate(([0],N0*np.array([2**i for i in range(p+1)])))
    seq_diff=seq[1:]-seq[:-1]
    return [seq,seq_diff]


def plevels(p,N0,l,l0,lz,T,z,collection_input):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=EnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
        else:
            [m,c]=EnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            xi= (1./seq[-1])*funct_val    
    if l_original>0:
        if p==0:
            [m,m1]=CEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(m[-1].T)[0]
            xi1=funct_val
            funct_val2[:,0]=(m1[-1].T)[0]
            xi2=funct_val2
            xi=xi1-xi2
        
        if p>0:
            
            for i in range(p+1):
                [m,m1]=CEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]             
                
            xi1= ((1./seq[-1])-(1./seq[-2]))*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]
            xi2= (1./seq[-1]-1./seq[-2])*np.sum(funct_val2[:,:-1],axis=1)+funct_val2[:,-1]/seq[-1]
            xi=(xi1-xi2).reshape((dim,1))
    return xi



def Dplevels(p,N0,l,l0,lz,T,z,collection_input):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=DEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
        else:
            [m,c]=DEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            xi= (1./seq[-1])*funct_val    
    if l_original>0:
        if p==0:
            [m,m1]=DCEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            funct_val[:,0]=(m[-1].T)[0]
            xi1=funct_val
            funct_val2[:,0]=(m1[-1].T)[0]
            xi2=funct_val2
            xi=xi1-xi2
        
        if p>0:
            
            for i in range(p+1):
                [m,m1]=DCEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]             
                
            xi1= ((1./seq[-1])-(1./seq[-2]))*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]
            xi2= (1./seq[-1]-1./seq[-2])*np.sum(funct_val2[:,:-1],axis=1)+funct_val2[:,-1]/seq[-1]
            xi=(xi1-xi2).reshape((dim,1))
    return xi


def phi2(x):
    return x

#Sampling function:

#This functions takes as argument an array representing the cumulative distribution of a random variable,and samples from it. 

#cumulative[0]=F(1)<cumulative[1]=F(2)<...<cumulative[n-1]=cumulative[-1]=F(n)=1.

def sampling(cumulative):
    u=np.random.uniform()
    for i in range(len(cumulative)):
        if cumulative[i]>u:
            p_sampled=i
            break
        
    return p_sampled

#Cumulatives

#Creates a pdf(and its cdf) $P(x)$ $x\in \{0,1,2,...,lmax\}$

#**Note:** To change the measure functions of the estimators it is necessary to change them directly into the functions single_term() and coupled_sum(). Don't forget that the functions have differents arguments. example: rh0.

def DF_l(lmax):
    array=np.arange(lmax+1)
    pdf=2**(-1.5*array)
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return pdf, cdf



def DF_p(lmax,l):
    if (lmax-l)<=4:
        array=np.arange(lmax-l+1)
        pdf=2**(4-array)
        pdf=pdf/np.sum(pdf)
        cdf= np.cumsum(pdf)
    if lmax-l>=5:
        array1=np.arange(5)
        array2=np.arange(5,lmax-l+1)
        pdf1=2**(4-array1)
        pdf2=array2*(np.log2(array2))**2/(2**(array2))
        pdf=np.concatenate((pdf1,pdf2))
        pdf=pdf/np.sum(pdf)
        cdf= np.cumsum(pdf)
    return [pdf,cdf]


def DF_l1(lmax,rho):
    array=np.arange(lmax+1)
    pdf=2**(-1*array*rho)
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return pdf, cdf

def DF_p1(pmax):
    array=np.array(range(pmax+1))
    pdf=(array+1)*(np.log2(array+2))**2/(2**(array))
    pdf=pdf/np.sum(pdf)
    cdf= np.cumsum(pdf)
    return [pdf,cdf]


#Single term estimator 
#B: Number of i.i.d. samples of the unbiased estimator

def single_term(B,N0,l0,lz,T,z,collection_input,pmax,lmax,rho):

    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    [pdf_l,cdf_l]=DF_l1(lmax,rho)
    #[pdf_l,cdf_l]=DF_l(lmax)
    for i in range(B):
        l=sampling(cdf_l)
        #[pdf_p,cdf_p]=DF_p1(pmax)
        #[pdf_p,cdf_p]=DF_p(lmax,l)
        [pdf_p,cdf_p]=DF_l1(pmax,rho)
        p=sampling(cdf_p)
        #print(i,l,p)
        print([p,l],i)
        xi=plevels(p,N0,l,l0,lz,T,z,collection_input)/(pdf_p[p]*pdf_l[l])
        estimators[i]=(xi.T)[0]
        pls[i]=[p,l]
    
    estimator_mean=np.mean(estimators,axis=0)
    #estimator_var=np.var(estimators,axis=0)

    return [estimator_mean,estimators,pls]

#Primary **test** for the single term estimator: Good 

#Linear regression function:

def coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    
    return np.asarray((b_0, b_1)) 

#Cost function depending on the levels

def cost_function(T,pls,N0):
    cost=0
    for i in range(len(np.array(pls).T[0])):
        if pls[i,1]==0:
            cost+=2**(pls[i,0]+pls[i,1])
        else:
            cost+=2**(pls[i,0])*(2**(pls[i,1])+2**(pls[i,1]-1))
    cost=cost*T*N0
    return cost

#In this section we are going to compute the coupled sum estimators in the number of samples level.
#The only two changes with respect the function plevels is that we have the additional arguments cdf_p and pdf_p, and that we are compututing $\Xi_{l,p}$ instead of $\Xi_{l,p}\mathbb{P}_P(p)$


def plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    N=seq[-1]
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=EnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
            xi=xi/pdf_p[p]
        else:
            [m,c]=EnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #xi= (1./seq[-1])*funct_val    
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            #funct_val[:,0]=np.sum(phi(x),axis=1)
            xi= (1./seq[-1])*funct_val/pdf_p[p]    
    if l_original>0:
        
        if p==0:
            #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[p]))
            #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[p]))
            #w1=np.cumsum(dw1,axis=0) 
            #w2=np.cumsum(dw2,axis=0) 
            #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
            #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
            [m,m1]=CEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #funct_val[:,0]=np.sum(phi(x),axis=1).T
            xi1= (1./seq[-1])*funct_val
            #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val2[:,0]=(seq_diff[0]*m1[-1].T)[0]
            xi2= (1./seq[-1])*funct_val2
            xi=xi1-xi2
        
        if p>0:
            mult_constants=np.array([(1/seq[i+1]-1/seq[i])/(1-cdf_p[i-1]) for i in range(1,p+1)])
            vector_constants=np.zeros(p)
            vector_constants2=seq[1:]*np.concatenate(([1],1-cdf_p[:p]))
            for i in range(p):
                vector_constants[i]=np.sum(mult_constants[i:])
            for i in range(p+1):
                #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[i]))
                #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[i]))
                #w1=np.cumsum(dw1,axis=0) 
                #w2=np.cumsum(dw2,axis=0) 
                #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
                #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
                [m,m1]=CEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
                #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[i],collection_input,w1,w2)
                #funct_val2[:,i]=np.sum(phi(x),axis=1)   
            term1= np.sum(vector_constants*(funct_val[:,:-1]-funct_val2[:,:-1]),axis=1)
            term2=np.sum((funct_val-funct_val2)/vector_constants2,axis=1)
            xi=(term1+term2).reshape(dim,1)
                
            
    return xi


def Dplevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p):
    [dim,dim_o]=collection_input[0:2]
    [seq,seq_diff]=nseq(p,N0)
    l_original=l
    l+=l0
    I=identity(dim).toarray()
    I_o=identity(dim_o).toarray()
    dt=2**(-l)
    N=seq[-1]
    
    funct_val=np.zeros((dim,p+1))
    funct_val2=np.zeros((dim,p+1))
    if l_original==0:
        if p>0:
            
            for i in range(p+1):
                [m,c]=DEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
            xi= ((1./seq[-1]-1./seq[-2])*np.sum(funct_val[:,:-1],axis=1)+funct_val[:,-1]/seq[-1]).reshape((dim,1))
            xi=xi/pdf_p[p]
        else:
            [m,c]=DEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #xi= (1./seq[-1])*funct_val    
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            #funct_val[:,0]=np.sum(phi(x),axis=1)
            xi= (1./seq[-1])*funct_val/pdf_p[p]    
    if l_original>0:
        
        if p==0:
            #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[p]))
            #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[p]))
            #w1=np.cumsum(dw1,axis=0) 
            #w2=np.cumsum(dw2,axis=0) 
            #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
            #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
            [m,m1]=DCEnKBF(T,l,lz,z,seq_diff[0],collection_input)
            #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val[:,0]=(seq_diff[0]*m[-1].T)[0]
            #funct_val[:,0]=np.sum(phi(x),axis=1).T
            xi1= (1./seq[-1])*funct_val
            #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[0],collection_input,w1[:,seq[p]:seq[p+1],:],w2[:,seq[p]:seq[p+1],:])
            funct_val2[:,0]=(seq_diff[0]*m1[-1].T)[0]
            xi2= (1./seq[-1])*funct_val2
            xi=xi1-xi2
        
        if p>0:
            mult_constants=np.array([(1/seq[i+1]-1/seq[i])/(1-cdf_p[i-1]) for i in range(1,p+1)])
            vector_constants=np.zeros(p)
            vector_constants2=seq[1:]*np.concatenate(([1],1-cdf_p[:p]))
            for i in range(p):
                vector_constants[i]=np.sum(mult_constants[i:])
            for i in range(p+1):
                #dw1=np.random.multivariate_normal(np.zeros(dim),dt*I,(T*2**(l),seq_diff[i]))
                #dw2=np.random.multivariate_normal(np.zeros(dim_o),dt*I_o,(T*2**(l),seq_diff[i]))
                #w1=np.cumsum(dw1,axis=0) 
                #w2=np.cumsum(dw2,axis=0) 
                #w1=np.concatenate(([w1[-1]-w1[-1]],w1),axis=0)
                #w2=np.concatenate(([w2[-1]-w2[-1]],w2),axis=0)
                [m,m1]=DCEnKBF(T,l,lz,z,seq_diff[i],collection_input)
                #[m,c,x]=EnKBF(T,l,lz,l,z,seq_diff[i],collection_input,w1,w2)
                funct_val[:,i]=(seq_diff[i]*m[-1].T)[0]
                funct_val2[:,i]=(seq_diff[i]*m1[-1].T)[0]
                #funct_val[:,i]=np.sum(phi(x),axis=1)
                #[m,c,x]=EnKBF(T,l-1,lz,l,z,seq_diff[i],collection_input,w1,w2)
                #funct_val2[:,i]=np.sum(phi(x),axis=1)   
            term1= np.sum(vector_constants*(funct_val[:,:-1]-funct_val2[:,:-1]),axis=1)
            term2=np.sum((funct_val-funct_val2)/vector_constants2,axis=1)
            xi=(term1+term2).reshape(dim,1)
                
            
    return xi



def coupled_sum(B,N0,l0,lz,T,z,collection_input,pmax,lmax):
    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    #[pdf_l,cdf_l]=DF_l(lmax)
    [pdf_l,cdf_l]=DF_l1(lmax,rho)
    for i in range(B):
        l=sampling(cdf_l)
        #[pdf_p,cdf_p]=DF_p1(pmax)
        #[pdf_p,cdf_p]=DF_p(lmax,l)
        [pdf_p,cdf_p]=DF_l1(pmax,rho)
        ##print(lmax,l,pdf_p)
        p=sampling(cdf_p)
        #print(i,l,p)
        xi=plevels_coupled(p,N0,l,l0,lz,T,z,collection_input,cdf_p,pdf_p)/pdf_l[l]
        #print(((xi.T)[0]))
        estimators[i]=(xi.T)[0]
        pls[i]=[p,l]
    
    estimators_mean=np.mean(estimators,axis=0)
    return [estimators_mean,estimators,pls]


# # Setting of parameters:
# 

# ### MSE in multiple dimensions

# The mse can be expressed as 
# 
# $$
# \begin{align}
# \operatorname{E}\left(\|\eta^{app}_t-\eta_t\|^2_2\right)=&\operatorname{E}\left(\|\eta^{app}_t-\overline{\eta^{app}_t}\|^2_2\right)+\|\overline{\eta^{app}}_t-\eta_t\|^2_2,
# \end{align}
# $$

# Control of the bias:
# 
# Let the bias of the EnKBF be 
# 
# $$
# \begin{align}
# \operatorname{E}\left(\eta^{N,l}_t-\eta_t\right)=&\operatorname{E}\left(\eta^{N,l}_t-\eta^l_t\right)+\operatorname{E}\left(\eta^{l}_t-\eta_t\right),\\
# &=k_1 \Delta_l+k_2\left(\frac{1}{N}\right),
# \end{align}
# $$

# We obtain a bias
# $$
# \begin{align}
# \operatorname{E}\left(\eta^{l}_t-\eta_t\right)
# =k_1 \Delta_l,
# \end{align}
# $$
# where $k_1$ is a vector of constants. 
# Now, using Richardson extrapolation
# 
# $$
# \begin{align}
# \operatorname{E}\left(\eta^{l}_t-\eta^{l-1}_t\right)
# =-k_1 \Delta_l,
# \end{align}
# $$
# if $\Delta_l=2^l$, which is precisely the case.  Taking the log of the absolute value
# $$
# \begin{align}
# \operatorname{Ln}|\operatorname{E}\left(\eta^{l}_t-\eta^{l-1}_t\right)^{(i)}|
# =\operatorname{Ln}|k_1^{(i)} \Delta_l|=\operatorname{Ln}|k_1^{(i)}|-l\operatorname{Ln}(2)=\operatorname{Ln}|k_1^{(i)}|+l\operatorname{Ln}(1/2).
# \end{align}
# $$
# for $i \in \{1,\dots,dim\}$

# ## F1

# In[18]:


#copy of the new one 
def fit_k1(seed_val,lmax,T,l0,N,collection_input,z,lz,REP):
    np.random.seed(seed_val)
    abscisas=np.array(range(l0,lmax))+1
    dim=collection_input[0]
    ordenadas=np.zeros((lmax-l0,dim))
    rep_array=np.zeros((REP,dim))
    for i in range(lmax-l0):
        l=l0+i+1
        lmaxE=l
        with progressbar.ProgressBar(max_value=REP) as bar:
            print("current level is",l,". ",lmax-l," remaining.")
            for rep in range(REP):
                [m,m1]=CEnKBF(T,l,lz,z,N,collection_input)
                rep_array[rep]=(m[-1,:,0]-m1[-1,:,0])
                bar.update(rep)
        ordenadas[i]=np.mean(rep_array,axis=0)
    bs=np.zeros((dim,2))
    for i in range(dim):
        bs[i]=coef(abscisas,np.log2(np.abs(ordenadas)).T[i])
    ordenadas=np.log2(np.abs(ordenadas))
    #b=coef(abscisas,ordenadas)
    x=abscisas
    y=ordenadas
    plt.plot(x,y)
    plt.plot(x,(bs[:,0].reshape(dim,1)+bs[:,1].reshape(dim,1)@x.reshape(1,lmax-l0)).T)
    
    return [2**bs,abscisas,ordenadas]


# Test and results

# Does the constant depend strongly on the model? 

# #### The infered bias in the discretization level is
# \begin{align}
# \operatorname{E}\left(\eta^{l}_t-\eta_t\right)
# =k_1 2^{-l}= 0.1377265*2^{-l}.
# \end{align}

# The bias in the number of samples is 
# $$
# \begin{align}
# \operatorname{E}\left(\eta^{N,l}_t-\eta^{l}_t\right)
# =k_2 \frac{1}{N},
# \end{align}
# $$
# Now, using Richardson extrapolation
# 
# $$
# \begin{align}
# \operatorname{E}\left(\eta^{N_p,l}_t-\eta^{N_{p-1},l}_t\right)
# =-k_2 \frac{1}{N_p},
# \end{align}
# $$
# if $N_p=2^p$, which can be the case.  Taking the log of the absolute value
# $$
# \begin{align}
# \operatorname{Ln}|\operatorname{E}\left(\eta^{N_p,l}_t-\eta^{N_{p-1},l}_t\right)|
# =\operatorname{Ln}|k_2 \Delta_l|=\operatorname{Ln}|k_2|-p\operatorname{Ln}(2)=\operatorname{Ln}|k_2|+p\operatorname{Ln}(1/2).
# \end{align}
# $$

# The inference of $k_2$ is in the sibling file "Unbiased estimator l-coupled sum"

# #### The infered bias in the number of samples is 
# \begin{align}
# \operatorname{E}\left(\eta^{N,l}_t-\eta^{l}_t\right)
# =k_2 \frac{1}{N}=0.09282754 \frac{1}{N}.
# \end{align}
# 

# Equating
# \begin{align}
#  bias^2=&\left\|\frac{k_2}{N}+k_1\Delta_l\right\|^2_2 \leq 2\left\|\frac{k_2}{N}\right\|^2_2+2\left\|k_1\Delta_l\right\|^2_2\\
#  =&\frac{\epsilon^2}{2} ,
# \end{align}
# then we set 
# 
# \begin{align}
# 2\left\|\frac{k_2}{N}\right\|^2_2=\frac{\epsilon^2}{4} ,\\
# 2\left\|k_1\Delta_l\right\|^2_2=\frac{\epsilon^2}{4} ,
# \end{align}
# 
# thus
# 
# \begin{align}
# p_{max}=\frac{1}{2}log_2\left(8 N_0^{-2} \epsilon^{-2}\|k_2\|^{2}_2 \right),\\
# l_{max}=\frac{1}{2}log_2\left( 8\epsilon^{-2}\|k_1\|^{2}_2 \right),
# \end{align}

# Variance sampling number of samples 

# In[19]:


def single_targetvariance(seed_val,MSE,N0,l0,T,collection_input,k1,k2,z,lz,rho):
    np.random.seed(seed_val)
    #[lmax,pmax]=par_lim(k1,k2,np.sqrt(MSE[-1]),N0)
    #[z,v]=gen_data(T,lmax,collection_input)
    costs=np.zeros(len(MSE))
    variances=np.zeros(len(MSE))
    dim=collection_input[0]
    means=np.zeros((len(MSE),dim))
    Bees=np.zeros(len(MSE))
    for i in range(len(MSE)):
        [lmax,pmax]=par_lim(k1,k2,np.sqrt(MSE[i]),N0)
        print([lmax,pmax])
        #print(type(lmax))
        [dim,dim_o]=collection_input[:2]
        #print(T,lmax,collection_input)
        #print(gen_data(T,lmax,collection_input))
        pdf_p=DF_l1(pmax,rho)[0]
        pdf_l=DF_l1(lmax,rho)[0]
        B=int(1/(pdf_p[-1]*pdf_l[-1]))
        [estimators,pls]=single_term(B,N0,l0,lz,T,z,collection_input,pmax,lmax,rho)[1:]
        variances[i]=np.sum(np.var(estimators,axis=0))
        Bees[i]=B
        means[i]=np.mean(estimators,axis=0)
        costs[i]=cost_function(T,pls,N0)
        while 2*variances[i]/B>MSE[i]:
            [estimators1,pls1]=single_term(B,N0,l0,lz,T,z,collection_input,pmax,lmax,rho)[1:]
            pls=np.concatenate((pls,pls1),axis=0)
            estimators=np.concatenate((estimators,estimators1),axis=0)
            variances[i]=np.sum(np.var(estimators,axis=0))
            costs[i]=cost_function(T,pls,N0)
            B+=B
            print("The variance/B is ",variances[i]/B,"and the mean cost is",costs[i]/B) 
            print("B is", B)
            means[i]=np.mean(estimators,axis=0)
            Bees[i]=B
    variances=variances/Bees
            
    return [variances,costs,means,Bees]


# ## F1 coupled sum
# 

# In[20]:


def coupled_targetvariance(seed_val,MSE,N0,l0,T,collection_input,k1,k2,z,lz):
    np.random.seed(seed_val)
    #[lmax,pmax]=par_lim(k1,k2,np.sqrt(MSE[-1]),N0)
    #[z,v]=gen_data(T,lmax,collection_input)
    costs=np.zeros(len(MSE))
    variances=np.zeros(len(MSE))
    dim=collection_input[0]
    means=np.zeros((len(MSE),dim))
    Bees=np.zeros(len(MSE))
    for i in range(len(MSE)):
        [lmax,pmax]=par_lim(k1,k2,np.sqrt(MSE[i]),N0)
        print([lmax,pmax])
        #print(type(lmax))
        [dim,dim_o]=collection_input[:2]
        #print(T,lmax,collection_input)
        #print(gen_data(T,lmax,collection_input))
        pdf_p=DF_l1(pmax,rho)[0]
        pdf_l=DF_l1(lmax,rho)[0]
        B=int(1/(pdf_p[-1]*pdf_l[-1]))
        [estimators,pls]=coupled_sum(B,N0,l0,lz,T,z,collection_input,pmax,lmax)[1:]
        variances[i]=np.sum(np.var(estimators,axis=0))
        Bees[i]=B
        means[i]=np.mean(estimators,axis=0)
        costs[i]=cost_function(T,pls,N0)
        while 2*variances[i]/B>MSE[i]:
            [estimators1,pls1]=coupled_sum(B,N0,l0,lz,T,z,collection_input,pmax,lmax)[1:]
            pls=np.concatenate((pls,pls1),axis=0)
            estimators=np.concatenate((estimators,estimators1),axis=0)
            variances[i]=np.sum(np.var(estimators,axis=0))
            costs[i]=cost_function(T,pls,N0)
            B+=B
            print("The variance/B is ",variances[i]/B,"and the mean cost is",costs[i]/B) 
            print("B is", B)
            means[i]=np.mean(estimators,axis=0)
            Bees[i]=B
    variances=variances/Bees
            
    return [variances,costs,means,Bees]


# In[21]:


def single_target(seed_val,epsilon,N0,l0,T,collection_input,DF_l1,DF_p1,k1,k2,rho):
    np.random.seed(seed_val)
    [lmax,pmax]=par_lim(k1,k2,epsilon,N0)
    #print(type(lmax))
    [dim,dim_o]=collection_input[:2]
    #print(T,lmax,collection_input)
    #print(gen_data(T,lmax,collection_input))
    [z,v]=gen_data(T,lmax,collection_input)
    B=100
    [estimators,pls]=single_term(B,N0,l0,lmax,T,z,collection_input,pmax,rho)[1:]
    var=np.sum(np.var(estimators,axis=0))
    print(var*2/B)
    while epsilon**2<var*2/B:
        [estimators1,pls1]=single_term(500,N0,l0,lmax,T,z,collection_input,pmax,rho)[1:]
        estimators=np.concatenate((estimators,estimators1),axis=0)
        pls=np.concatenate((pls,pls1),axis=0)
        var=np.sum(np.var(estimators,axis=0))
        B+=500
        print(var/B)
    estimator_mean=np.mean(estimators,axis=0)
    return [estimator_mean,estimators,pls]


# In[ ]:




