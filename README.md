# Unbiased-EnKBF
Code used to produce unbiased estimators of the Ensemble Kalman-Bucy Filter
This code is composed by several files that are used to generate the observations, optimizing constants, parallelized unbiased estimators and parallelized MLEnKBF, second moments relation and bias relations, many of the files serve for general purpose.

I list the necessary files to generate each quantity

Observations and signal:

(D)MLEnKBF-Parameter Tuning.ipynb

Optimization constants or parameter tuning 
    for c2 and c3
    (D)MLEnKBF-Parameter Tuning.ipynb
    for k1 and k2
    here is important to notice that I had to create a new EnKBF to compute k2 given that the whole realization of the brownian motions in time had
    to be stored
    DUnbiased EnKBF l-coupled sum.ipynb

python files with the necessary functions

DUnbiased_l_functions.py

Parallelized versions of the MLEnKBF

DMLEnKBF.py
MLEnKBF.py

Parallelized versions of the EnKBF
Unbiased500dim.py
DUnbiased500dim.py


Plot generations

Plots.ipynb
Plots_clean.ipynb
    






