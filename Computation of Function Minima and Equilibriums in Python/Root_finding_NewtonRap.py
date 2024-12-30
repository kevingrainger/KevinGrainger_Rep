# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:47:28 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pylab as plt
x=np.arange(0.1,5,0.01)
#our x range
#our function
A=1090
p=0.033
y=1.44
def f(x):
    return A*(np.exp(-x/p))-(y/(x))

                             #function derivative
def derivf(x):
    return -A/(p)*((np.exp(-x/p)))-(y/(((x**2))))

def ddf(x):
    return A/(p**2)*((np.exp(-x/p)))-(2*y/(x**3))
#Printing our function and the derivative
def functionabrv(x):
    return derivf(x)/ddf(x) # subs to make our equations neat

print (f(x))
plt.plot(x,f(x))
plt.plot(x,derivf(x))
plt.ylim(-40,40)
plt.xlim(0,1)
tol=0.0000000001 #Tolerance
x1=0.5 #random number in range
#shows the mechanism 
#Our Newton Rampson function
def NR(x): 
   x1=0.1
   while abs(derivf(x)) >= tol: #runs while x is greater than 0.0001
        x = x - functionabrv(x) #Newton Rampson definition
        #keep track of the steps taken
        #plt.scatter(x,derivf(x))
        #return x, nsteps  
        return x
NR(x1)
print (NR(x1)) #our root
print (f(x1))
#plt.scatter(x1,f(x1),color='red')
plt.plot(NR(x1),f(x1),'ro')
plt.xlabel('x position')
plt.ylabel('Respective F(x)')
plt.show()