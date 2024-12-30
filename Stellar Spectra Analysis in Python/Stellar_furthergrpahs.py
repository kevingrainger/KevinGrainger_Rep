# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:15:42 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pyplot as plt


T=6346 #kelvin
h=6.6260e-34
k= 1.3803-23
c= 2.99e+8 #in args maybe i dunno

wave=np.arange(1e-8,2e-6,1e-8)

a = (2.0*h*c**2)/(wave**5)
b = h*c/(wave*k*T)
y= a/((np.exp(b)-1.0))

plt.plot(wave,np.exp(b))
#def planck(wave,T):
    #a = 2.0*h*c**2
    #b = h*c/(wave*k*T)
    #y= a/ ((wave**5) * (np.exp(b) - 1.0) )
    #return y

#plt.plot(wave,planck)