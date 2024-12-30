# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:19:20 2022

@author: Kevin Grainger
"""

import matplotlib.pyplot as plt
import math
#from scipy.integrate import simp
import numpy as np

#our starting values 
m= 1.0 #Part_mass
M = 100.0 #piston mass
T=1.0
F= -10.0
N=10        #number of particles
interactions= 10*N
KB=1.38e-23 
t=0         #overall time variable
#For these initial conditions and velocities we can calculate the equilibrium
#via the ideal gas law, as detailed in the pdf
#to start we need to define our arrays that will store any data
#I prefer 1D arrays as the code is easier
#I will use the method of defining arrays as in the last coding exercise
#As it was cleaner
x=[]           #Particle postion
X=[]
print(X)          #piston postion
time=[]    #time array- will probable we changed to an np.arange
H=[]    #enpathly for later question
v=[]
V=[]
X.append(0.005)
print(X) 
#As in the tutorial we need to start by defining our initial particle energies
#This is derived in the pdf file
U=np.sqrt((KB*T/m)+(M*(X[0]**2)/m*N)) #initial energy
print(U)
V[0]=0.1
X[0]=np.random.normal(0,1)
#as calculated in the pdf we can say our particle velocities can 
#be choosen from the following normal distrution with standev U
#We known from the coding exercises to use the np.random.choice()
# to repesent whether the particles are moving towards or away
#from the RHS piston ie.[1,-1]
for i in range (N):
    v=(np.random.normal(U,U/N)*np.random.choice([1,-1]))

for i in range(N):
    x[i]=np.random.normal(0,X[0])
V_int=(np.random.normal(U,U/N)) 

print (X)
#To begin we need an expression for piston Velocity
#First becasue we need to graph it's progression
#But also as the expressions for particle velocity etc depend on it
#As defined in the pdf:
#defining a small time step 
step=0.1
#we have to figure out which paticle hits the piston first, 
#the one travelling left or right
#the below exercise is only there to determine that
#The wait time function
def wait_time(x,v,X,V):
    tou=(M/F)*(V-v+np.sqrt((V-v)**2-2*(F/M)*(x-X)))
    return tou
def wait_time_neg(x,v,X,V):  
    tou_neg=(M/F)*(V+v+np.sqrt((V+v)**2-2*(F/M)*(x+X)))

    return tou_neg
a=1  #random constant to keep track of partilce direction
for i in range(N):
    x1=x[i]
    v1=v[i]
    
    wait_time(x,v,X,V)
                        
    wait_time_neg(x,v,X,V)
    if wait_time(x, v, X, V)<wait_time_neg(x, v, X, V):
        time=wait_time(x, v, X, V)
        a=1
    else:
        time=wait_time_neg(x, v, X, V)
        a=-1
T+=time
      
#we want the simulation to stop at some point so we set a large time limit;
#now it's time to update the EOM of he constiuents:
  
#Particle EOM
for i in range (N):
    x[i]=x[i-1]+v[i]*t

#for the aul piston
 #eom as in the pdf, not in for loop as it's dependant on all particles at once
X[0]+=X[0]+V*(time)+(0.5)*(F/M)*(t**2)
  #we can track it continously over time
#finally we need to know if the particle is goin left or right
#so we can update all the velocities accordingly
#We use the equations given in Q1 (e)
#;I will us the coeff a to replace the plus minus sign
W=((2*m*V)/m+M) + (M-m)*V/(m+M)    #piston velocity
w=((2*m*v)/m+M) + (m-M)*v/(m+M)     #partilce velocity

if a==1:
   W=((2*m*V)/m+M) + (M-m)*V/(m+M)
   w=((2*m*v)/m+M) + (m-M)*v/(m+M) 
else:
   W=((-2*m*V)/m+M) + (M-m)*V/(m+M)
   w=((-2*m*v)/m+M) + (m-M)*v/(m+M)

time.append(t)

plt.plot(T,X)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Piston progression over Time (X vs. T)')  
plt.plot(T,U)

    
    
    
    
    
    
    
    
    
    