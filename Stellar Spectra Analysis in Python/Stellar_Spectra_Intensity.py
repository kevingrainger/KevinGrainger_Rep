# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:57:43 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pyplot as plt


T=6346 #kelvin
h=6.626*(10**-12)
k= 1.38*(10**-10)
c= 2997924*(10**10) #in args ps

#x=np.arange(100,20000,1)     ##now in meters 
x=np.linspace(100,2000,1)
W=np.linspace(100,20000,1)
def pfunction(W):
    A=2.0*h*c**2 #add pi back in
    B=(h*c)/((W*k*T))
    Intensity=  A/((W**5)*(np.exp(B)-1.0))
    return Intensity 
#L=1/(x**5)
#y=A*C
#y=((2*np.pi*h*(c**2))/(x**5))*(1/((np.e**(B))-1))


plt.plot(W,pfunction(W))
plt.show()
 