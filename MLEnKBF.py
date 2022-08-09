#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:40:42 2022

@author: alvarem
"""



from Functions_file_UEnKBF import *
import multiprocessing
import time
import collections

#In this part we define the "arguments" that will go in the Psingle_term\
#function, and define the Psingle_term function. REmember taht this part will 
#be uploaded by all the processes.

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


samples=500
blocks=50
l0=4
N0=50
C=1
pmax=5
lmax=7
rho=999./1000.

def cost_mlenkbf(C,T,L,l0):
    num=Numcal(C,L,l0,w1,w2)
    kk=num[0]*2**(l0)
    for l in range(l0,L):
        kk=kk+num[l-l0]*(2**(l+1))*3/2.
    return(T*kk)


def Numcal(C,L,l0,w1,w2):
    Num=np.zeros(L-l0+1,dtype=int)
    epsilon = np.sqrt(2)*c2 * (1./2.)**(L)/np.sqrt(w1)
    if L-l0>0:
        Num[0] =np.maximum(np.ceil(4*c1 * epsilon**(-2)/w2),2)
    else: 
        Num[0] =np.maximum(np.ceil(4*c1 * epsilon**(-2)/w2),2)/2
    for li in range(1,L-l0+1):
        l=l0+li
        Num[li]=np.maximum(np.ceil(2*epsilon**(-2)*c3*(L-l0)*2**(-l))/w2,2)
        
    return(np.ceil(Num*C))

def MLEnKBF(C,T,L,lz,l0,z,collection_input):
    Num=Numcal(C,L,l0,w1,w2)
    telescopic_summand=np.zeros((L-l0+1,dim))
    [m,c]=EnKBF(T,l0,lz,z,int(Num[0]),collection_input)
    telescopic_summand[0]=m[-1,:,0]
    for l in range(l0+1,L+1):
        [m1,m2]=CEnKBF(T,l,lz,z,int(Num[l-l0]),collection_input)
        telescopic_summand[l-l0]=m1[-1,:,0]-m2[-1,:,0]
    est=np.sum(telescopic_summand,axis=0)
    return est

def Psimulate_mse(arg_col):
    [seed_val,collection_input,z]=arg_col
    print("the block is", seed_val)
    
    #lmax = Lmax
    np.random.seed(seed_val)
    dim=collection_input[0]
    collection_input = gen_model(dim,amm,bmm,cmm)
    
    #np.random.seed(seed_val_obs)
    #z = gen_data(T,lmax,collection_input)[0]
    zshape=(163841, 10, 1)
    #vshape=(163841, 10, 1)
    zload=np.loadtxt('obsT80d100'+str(seed_val)+'.txt', dtype=float)
    #vload=np.loadtxt('obsT80d100'+str(block_sh)+'.txt', dtype=float)
    z=zload.reshape(zshape)
    #v=vload.reshape(vshape)
    lkbf=11
    tv_approx=np.reshape(KBF(T,lkbf,lz,z,collection_input)[0][-1],dim)
    
    mse_ml=np.zeros((L-l0+1,dim))
    levels=np.array(range(l0,L+1))
    mean_ml=np.zeros((L-l0+1,dim))
    delt_level=np.zeros(10)

    ML_real_levels=np.zeros((L-l0+1,Rep,dim))
    costs=np.zeros(L-l0+1)
    for l in range(l0,L+1):
        est_ml=np.zeros((Rep,dim))
        #est_en=np.zeros(Rep)
        print("Current level is",l,".",L-l, "remaining.")
        seed_j=0
        with progressbar.ProgressBar(max_value=Rep) as bar:
            ML_real=np.zeros((Rep,dim))
            for rep in range(Rep):
                with np.errstate(divide='ignore'):
                    #mean_en = DEnKBF(T,l,lmax,z,N,collection_input)[0][-1,:]
                    #print(T,l)
                    np.random.seed(seed_j)
                    ML_real[rep] = MLEnKBF(C,T,l,lz,l0,z,collection_input)
                    seed_j+=1
                    est_ml[rep] = (ML_real[rep] - tv_approx)**2
                    #est_en[rep] = np.sum((mean_en - tv_approx)**2)
                    bar.update(rep)
            mean_ml[l-l0]=np.mean(ML_real,axis=0)       
            mse_ml[l-l0]=np.mean(est_ml,axis=0)
            costs[l-l0]=cost_mlenkbf(C,T,l,l0)
            ML_real_levels[l-l0]=ML_real
            
        #print(mse_ml[l-l0])
    
    return [ML_real_levels,costs,tv_approx]


T=80
dim=10
#seed_val_col=5
#seed_val_obs=5
amm=2.
bmm=2.
cmm=50.
b=6
lz=11
#l=4
#N=50
#tau=1/2**(l)
#J=T*(2**l)
#seed_val=2

c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=279/280.
w1=1./280.


Rep=100
l0=6
L=6
blocks=30

if __name__ == '__main__':
    start=time.time()
    
    
    for l in range(l0,L+1):
        print(Numcal(C,l,l0,w1,w2),cost_mlenkbf(C,T,l,l0))
    #definition of the arrays that will contain all the results
    
    #We take each one of the blocks and compute in parallel
    inputs=[]
    for block_sh in range(blocks):
        block_sh+=20
        np.random.seed(block_sh)
        collection_input=gen_model(dim,amm,bmm,cmm)
        zshape=(163841, 10, 1)
        vshape=(163841, 10, 1)
        zload=np.loadtxt('obsT80d100'+str(block_sh)+'.txt', dtype=float)
        #vload=np.loadtxt('realiT80d100'+str(block_sh)+'.txt', dtype=float)
        z=zload.reshape(zshape)
        #v=vload.reshape(vshape)
        inputs.append([block_sh,collection_input,z])
        
    pool = multiprocessing.Pool(processes=10)
    pool_outputs=pool.map(Psimulate_mse, inputs)
        #Copy the results into new arrays
     
    pool.close()
    pool.join() 
    
    
       
    ML_real_levels=np.zeros((L-l0+1,Rep,blocks*dim))
    
    for block in range(blocks):
        
        ML_real_levels[:,:,block*dim:(block+1)*(dim)]=pool_outputs[block][0]
        MLrl_reshaped=ML_real_levels.reshape((L-l0+1)*Rep*blocks*dim)
        
    np.savetxt("_MLEnKBF_3nd_nn.3or.txt",MLrl_reshaped ,fmt="%f")

    end=time.time()
    #print(blocks_pools)
    print("Parallelized processes time:",end-start,"\n")
    #print("Pool outputs are",pool_outputs)

"""
For _MLEnKBF_3nd_nn.3or.txt for the blocks from 20 to 39
#seed_val=2

c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=279/280.
w1=1./280.

Rep=100
l0=6
L=6
blocks=30




MLEnKBF_3nd_nn.1or.txt from blocks 10 to 19
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=279/280.
w1=1./280.


Rep=50
l0=6
L=6





MLEnKBF_3nd_nn.2.txt from blocks 0 to 9
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=279/280.
w1=1./280.
Rep=100
l0=6
L=9





MLEnKBF_3nd_nn.1.txt from blocks 10 to 19
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=279/280.
w1=1./280.


Rep=50
l0=6
L=9


MLEnKBF_2nd_nn.2.txt from blocks 10 to 29

c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=399./400.
w1=1./400.
Rep=100
l0=6
L=9
blocks=10




MLEnKBF_2nd_nn.1.txt fro the fist 10 blocks

c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=399./400.
w1=1./400.
Rep=100
l0=6
L=9
blocks=10



MLEnKBF_2nd_test
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=49./50.
w1=1./50.

Rep=3
l0=5
L=8
blocks=10


For MLEnKBF_nn.4.txt for the blocks from 40 to 49
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=29./30.
w1=1./30.

Rep=100
l0=4
L=7
blocks=30



For MLEnKBF_nn.3.txt for the blocks from 10 to 39
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=29./30.
w1=1./30.

Rep=100
l0=4
L=7
blocks=30




For MLEnKBF_nn.1.txt we have the first 10 blocks
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=29./30.
w1=1./30.

Rep=100
l0=4
L=7
blocks=10



For test_MLEnKBF_nn.txt
c1=227601.124639
c2=665.4957501794402
c3= 7949954.812580001
w2=29./30.
w1=1./30.

Rep=3
l0=4
L=6
blocks=4


MLENKBF_10blocks_50samples.txt corresponds to the first 50 samples of 
the blocks from 0 to 9 with parameters 

c1=145863.012072
c2=623.8111694025733
c3= 237040.39353300002
w2=49./50.
w1=1./50.

Rep=50
l0=4
L=9
blocks=10






"ML500b.txt" corresponds to the Multilevel algorithm for the efirst 10 block
with parameters and estimatd constatns 
c1=239308.954479 
c3=1012107.5286470001
c2= 665.4957501794402
w2=349./350.
w1=1./350.

Rep=100
N=50
l0=4
L=9



"ML500b_a.txt" corresponds to the Multilevel algorithm for the blocks from
10 to 39with parameters and estimatd constatns 
c1=239308.954479 
c3=1012107.5286470001
c2= 665.4957501794402
w2=349./350.
w1=1./350.

Rep=100
N=50
l0=4
L=9




"ML500b_b.txt" corresponds to the Multilevel algorithm for the blocks from
40 to 79 with parameters and estimatd constatns 
c1=239308.954479 
c3=1012107.5286470001
c2= 665.4957501794402
w2=349./350.
w1=1./350.

Rep=100
N=50
l0=4
L=9



"ML500b_r.txt" corresponds to the Multilevel algorithm for the efirst 40 blocks
with parameters and estimatd constatns 
c1=239308.954479 
c3=1012107.5286470001
c2= 665.4957501794402
w2=349./350.
w1=1./350.

Rep=80
N=50
l0=4
L=8
the shape of MLrl_reshaped is (L-l0+1,Rep,blocks*dim)






"ML500b_r.txt" corresponds to the Multilevel algorithm for the efirst the blocks from
40 to 79  and estimatd constatns 
c1=239308.954479 
c3=1012107.5286470001
c2= 665.4957501794402
w2=349./350.
w1=1./350.

Rep=80
N=50
l0=4
L=8
the shape of MLrl_reshaped is (L-l0+1,Rep,blocks*dim)


#This part of code is made to upload the files saved in the previous section 


estimators_shape=(blocks, samples, dim)
pls_shape=(blocks, samples, 2)
estimators_load=np.loadtxt('estimators_whole_T80d10.txt', dtype=float)
pls_load=np.loadtxt('pls_whole_T80d10.txt', dtype=float)
estimators_whole=estimators_load.reshape(estimators_shape)
pls_whole=pls_load.reshape(pls_shape)
print(estimators_whole ,"\n")
print(pls_whole ,"\n")



#"""


"""


start=time.time()
estimatorssamples=np.zeros((samples,dim))
plssamples=np.zeros((samples,2))
for i in range(samples):
    np.random.seed(i)
    [estimator_mean,estimatorssamples[i],plssamples[i]]=single_term(B,N0,l0,lz,T,z,collection_input,pmax,lmax,rho)


end=time.time() 
print("Non-Parallelized processes time:",end-start) 
#print(plssamples)


#"""
