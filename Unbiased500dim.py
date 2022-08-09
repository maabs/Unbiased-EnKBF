    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:40:42 2022

@author: alvarem
"""

#Upload the information and the packages 

from Functions_file_UEnKBF import *
import multiprocessing
import time
import collections
T=80
dim=10
seed_val_col=5
seed_val_obs=5
amm=2.
bmm=2.
cmm=50.
b=6
lz=11
l=4
N=50
tau=1/2**(l)
J=T*(2**l)
seed_val=2


"""
np.random.seed(seed_val_col)
collection_input=gen_model(dim,amm,bmm,cmm)
#np.random.seed(seed_val_obs)
#[z,v] = gen_data(T,lz,collection_input)
#[m,c] = KBF(T,l,lmax,z,collection_input)
#ems=np.zeros((B,J+1,dim,1))

zshape=(163841, 10, 1)
vshape=(163841, 10, 1)
zload=np.loadtxt('obsT80d100.txt', dtype=float)
vload=np.loadtxt('realiT80d100.txt', dtype=float)
z=zload.reshape(zshape)
v=vload.reshape(vshape)

"""





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


def Pcoupled_sum(arg_col):
    
    
    [seed_val,collection_input,z]=arg_col
    np.random.seed(seed_val)
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
    return [estimators,pls]

    
def Psingle_term(arg_col):
    [seed_val,collection_input,z]=arg_col
    np.random.seed(seed_val)
    [dim,dim_o]=collection_input[0:2]
    estimators=np.zeros((B,dim))
    pls=np.zeros((B,2))
    estimators_sum=0
    [pdf_l,cdf_l]=DF_l1(lmax-l0,rho)
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

    return [estimators,pls]


samples=5
blocks=2
l0=4
N0=50
B=1
pmax=5
lmax=7
rho=999./1000.
#"""
start=time.time()




if __name__ == '__main__':
    blocks_pools=[]
    
    for block in range(blocks):
        np.random.seed(block)
        collection_input=gen_model(dim,amm,bmm,cmm)
        zshape=(163841, 10, 1)
        vshape=(163841, 10, 1)
        zload=np.loadtxt('obsT80d100'+str(block)+'.txt', dtype=float)
        vload=np.loadtxt('obsT80d100'+str(block)+'.txt', dtype=float)
        z=zload.reshape(zshape)
        v=vload.reshape(vshape)
        inputs = [[i+1000,collection_input,z] for i in range(samples)]
        pool = multiprocessing.Pool(processes=8)
        pool_outputs = pool.map(Psingle_term, inputs)
        pool.close() 
        pool.join() 
        blocks_pools.append(pool_outputs)
        
        #blocks_pools[block]=pool_outputs
    #print ('Pool    :', pool_outputs)
    end=time.time()
    print(blocks_pools)
    print("Parallelized processes time:",end-start,"\n")

    estimators=np.zeros((blocks,samples,dim))    
    for block in range(blocks):
        for rep in range(samples):
            estimators[block,rep]=blocks_pools[block][rep][0]
    estimators=estimators.flatten()
    print(estimators)
    np.savetxt("test_Unbiased.txt",estimators,fmt="%f")
    
    blocks_pools=[]
    start1=0
    end1=0
    
    for block in range(blocks):
        print("Current block is ",block)
        print("last block lasted ",end1 -start1)
        start1=time.time()
        np.random.seed(block)
        collection_input=gen_model(dim,amm,bmm,cmm)
        zshape=(163841, 10, 1)
        vshape=(163841, 10, 1)
        zload=np.loadtxt('obsT80d100'+str(block)+'.txt', dtype=float)
        vload=np.loadtxt('obsT80d100'+str(block)+'.txt', dtype=float)
        z=zload.reshape(zshape)
        v=vload.reshape(vshape)
        inputs = [[i+1000,collection_input,z] for i in range(samples)]
        pool = multiprocessing.Pool(processes=8)
        pool_outputs = pool.map(Pcoupled_sum, inputs)
        pool.close() 
        pool.join() 
        blocks_pools.append(pool_outputs)
        end1=time.time()
        
        
        #blocks_pools[block]=pool_outputs
    #print ('Pool    :', pool_outputs)
    end=time.time()
    print(blocks_pools)
    print("Parallelized processes time:",end-start,"\n")

    estimators=np.zeros((blocks,samples,dim))    
    for block in range(blocks):
        for rep in range(samples):
            estimators[block,rep]=blocks_pools[block][rep][0]
    estimators=estimators.flatten()
    print(estimators)
    np.savetxt("Unbiased_dim500CS.txt",estimators,fmt="%f")
    
    
    
    #print("Pool outputs are",pool_outputs)
