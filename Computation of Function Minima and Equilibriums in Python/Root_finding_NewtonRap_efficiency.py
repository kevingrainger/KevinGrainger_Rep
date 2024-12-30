# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 03:03:15 2022

@author: Kevin Grainger
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:39:05 2022

@author: Kevin Grainger
"""
import numpy as np
import matplotlib.pylab as plt
x=np.arange(-10,10,0.1)
#our x range
#our function
def f(x):
    return x**2-x*3-5
#function derivative
def derivf(x):
    return 2*x-3
#Printing our function and the derivative
print (f(x))
plt.plot(x,f(x))
plt.plot(x,derivf(x))
plt.plot(x,0*f(x))
plt.xlabel('x')
plt.ylabel('f(x)')
nsteps=0
tol=0.0001 #Tolerance
x0=10 # This chose a number far from the root so that the graph clearly
#shows the mechanism 
#Our Newton Rampson function
def NR(x,nsteps):
    nsteps=0 
    while abs(f(x)) >= tol: #runs while x is greater than 0.0001
        x = x - f(x)/derivf(x) #Newton Rampson definition
        nsteps+=1 #keep track of the steps taken
        plt.scatter(x,f(x))
    return x, nsteps   
NR(x0,nsteps) #calling the function
print (NR(x0,nsteps)) #our root
plt.show()


tol = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001,0.000005, 0.000002, 0.000001]
n = 0
steps = 0

def nsteps(x,nsteps):
    n = 0
    while n <14:
        while np.abs(f(x))> tol[n]:
            x = x - f(x)/derivf(x)
            nsteps+=1 #keep track of the steps taken
            plt.scatter(x,f(x))
            return x, nsteps
        plt.scatter(np.log10(tol[n]),steps)
        n += 1
nsteps(x,nsteps)
plt.xlabel("Tolerance^10")
plt.ylabel("No. of Steps")
plt.show()