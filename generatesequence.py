import math
import numpy as np
from astropy import constants as ast_const
import matplotlib
#matplotlib.use('TkAgg') # needed for Yilda to make matplotlib work in virtualenv
import matplotlib.pyplot as plt

# seaborn style file
#import seaborn as sns; sns.set()
#import seabornstyle

from functions import rksolver
from functions import scaleplots

#define constant
r_sun = ast_const.R_sun.value

# given a number, creates a set of n main sequence stars and returns
# their scaled luminosities, masses, radii and surface temperatures
def mainsequence(n):
    #empty arrays to store star surface data
    T = []
    L = []
    M = []
    R = []
    
    # generate range of radii from 0.01 R_sun to 1000 R_sun 
    radius_range = np.random.randint(0.01, 1000, n)
    
    for rad in radius_range:
        #produce data for star
        (r_surf, M_vals, rho_vals, T_vals, L_vals) = rksolver(rad*r_sun);
        #scale data to dimensionless units
        (norm_M, norm_rho, norm_L, norm_T) = scaleplots(M_vals, rho_vals, T_vals, L_vals);
        T.append(T_vals[-1])
        L.append(norm_L[-1])
        M.append(norm_M[-1])
        R.append(rad)
        print(R)
        print(T)
        print(L)
    return (R, T, L, M)

#example: create main sequence of 20 stars and plot HR diagram            
(rad, temp, lum, mass) = mainsequence(20)

plt.scatter(temp, lum);
plt.gca().invert_xaxis()
plt.show()