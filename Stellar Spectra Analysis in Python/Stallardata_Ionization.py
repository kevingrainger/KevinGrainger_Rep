# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 08:09:27 2022

@author: Kevin Grainger
"""

import matplotlib.pyplot as plt
import math
#from scipy.integrate import simp
import numpy as np

n_e=46e-19  #electron density
L=1.224e-9      #broglie 
g_1=2
g_2=8           #degenergices as before
K=1.38e-23      #boltz constant

T=np.arange(2000,90000,100)     #temp range
  
Coeff=(g_2/g_1)             #ratio of degeneracy
print(Coeff)
for i in T:         #looping over temps
    F=(Coeff)*np.e**(-(10.4*1.6e-19)/(K*T))     #boltzmann eq.
print (F)
plt.plot(T,F)
plt.ylabel('Fraction Ionised')
plt.xlabel('Temp')

