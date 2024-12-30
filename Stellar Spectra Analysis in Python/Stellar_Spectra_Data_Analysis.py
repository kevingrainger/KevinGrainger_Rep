# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:07:52 2022

@author: Kevin Grainger
"""

import numpy as np
import matplotlib.pyplot as plt

h= 6.62606957e-34
c= 299792458
k= 1.3806488e-23
T=6346

w = np.linspace(100, 20000, 4000)
#F = 2 * h * c**2 / (w*1.e-9)**5 / (np.exp(h * c / w/1.e-9 / k / T) - 1)
A=2*h*(c**2)
W=((w*1.e-10)**5)
B=(np.exp(h*c/(w*1.e-10)/k/T)-1)
PF=(A/W/B)
plt.plot(w,PF)