# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 03:15:22 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pylab as plt
x=np.arange(0.1,1,0.001)
k=1.44
A=1090
p=0.033
l=1.577122
#l=(Ae**1/p)
def f(x):
    return l*((np.e)**-x)-(1/x)*k

def derivf(x):
   return l*((np.e)**-x)-(1/(x**2))*k

f(x)
derivf(x)

plt.show()

nsteps=0
tol=0.0001 
x1=0.1
#Tolerance
 # This chose a number far from the root so that the graph clearly
#shows the mechanism 
#Our Newton Rampson function
def NR(x):
    while abs(f(x)) >= tol: #runs while x is greater than 0.0001
        x = x - f(x)/derivf(x) #Newton Rampson definition
        plt.scatter(x,f(x))
    return x   
NR(x1) #calling the function
print (NR(x)) #our root

#tol=0.001
#def NR(x):
 #   while abs(F(x)) >= tol: #runs while x is greater than 0.0001
  #      x = x - F(x)/dervF(x) #Newton Rampson definition
   #     plt.scatter(x,F(x))
    #return x   
#NR(x) #calling the function
#print (NR(x)) #our root
plt.plot(x,f(x))
plt.plot(x,derivf(x))
plt.ylim(10,-60)
plt.scatter()
plt.show()