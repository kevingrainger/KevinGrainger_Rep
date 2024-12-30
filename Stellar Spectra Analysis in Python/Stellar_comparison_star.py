# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:48:58 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pyplot as plt

h= 6.62606957e-27 #erg s
c= 2.99792458e18 #angstrom/s
k= 1.3806488e-16 #erg/K
T1,T2=6346,5776 #K

w = np.linspace(100, 20000, 4000) #Creates our range of wavelenghts

#in the following lines I have split Planck's Function into parts
#A (the Numerator) ,W (referring to wavelenght), b (to simplify the exponent on e) 
#finally B (the denominator)
A=2*h*(c**2)
W=(w**5)
b1=h*c/(w*k*T1) #Because in this graph we have two spectra I have -
b2=h*c/(w*k*T2) #split the parts of the function into parts with -
B1=(np.exp(b1)-1)#the different temperatures T1&T2
B2=(np.exp(b2)-1)
PF1=(A/W/B1) #A sperate function for each star
PF2=(A/W/B2)

plt.xlabel('Wavelenght-(Angstrom)') #our labels
plt.ylabel('F(λ)=erg/s/Angstrom^-3')
plt.plot(w,PF1,label="Mystar")
plt.plot(w,PF2,label="The Sun")
plt.axvline(x = 4000, color = 'r')
plt.axvline(x = 7500, color = 'b')
plt.legend(loc="upper right")
plt.show()


