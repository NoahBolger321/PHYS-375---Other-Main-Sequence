import math
import numpy as np
from astropy import constants as ast_const
import matplotlib
#matplotlib.use('TkAgg') # needed for Yilda to make matplotlib work in virtualenv
import matplotlib.pyplot as plt
import sys

# seaborn style file
#import seaborn as sns; sns.set()
#import seabornstyle

# defining constants
h_bar = ast_const.hbar.value
G = ast_const.G.value
c = ast_const.c.value
m_e = ast_const.m_e.value
m_p = ast_const.m_p.value
k_b = ast_const.k_B.value
r_sun = ast_const.R_sun.value
m_sun = ast_const.M_sun.value
l_sun = ast_const.L_sun.value
sigma_SB = ast_const.sigma_sb.value
gamma = 5 / 3.0
a = 4 * sigma_SB / c

# mass fractions
X = 0.7
Y = 0.28
Z = 0.02
mew_mp = (1 / ((X) * 2 + 0.75 * (Y) + 0.5 * (Z))) * m_p

# units rho_c: [kg/m**3]
rho_c = 162200
# units T_c: [K]
T_c = 15710000


# function returning Pressure
def P(rho, T):
    first_term = ((3 * math.pi ** 2) ** (2 / 3.0) / 5.0) * ((h_bar ** 2) / m_e) * (rho / m_p) ** (5 / 3.0)
    second_term = rho * (k_b * T) / mew_mp
    third_term = (1 / 3.0) * a * T ** 4
    return first_term + second_term + third_term


# DE for pressure in terms of rho
def dP_drho(rho, T):
    first = (((3 * math.pi ** 2) ** (2 / 3.0)) / 3.0) * ((h_bar ** 2) / (m_e * m_p)) * (rho / m_p) ** (2 / 3.0)
    second = k_b * T / mew_mp
    return first + second


# DE for pressure in terms of temperature
def dP_dT(rho, T):
    return rho * k_b / (mew_mp) + (4 / 3.0) * a * (T ** 3)  # fixed - CM


# DE for mass in terms of radius
def dM_dR(rho, rad):
    return (4 * math.pi * rad ** 2) * rho


# DE for luminosity in terms of radius
def dL_dR(rho, rad, eps):
    return (4 * math.pi * rad ** 2) * rho * eps


# DE for tau in terms of radius
def dTau_dR(kappa, rho):
    return kappa * rho


# DE for density in terms of radius
def drho_dR(rho, rad, mass, dPT, dTR, dPrho):
    num = -((G * mass * rho / (rad ** 2)) + dPT * dTR)
    denom = dPrho
    return num / denom


# DE for temperture in terms of radius
def dT_dR(kappa, rho, rad, T, lum, mass, press):
    first = 3 * kappa * rho * lum / (16 * math.pi * a * c * (T ** 3) * (rad ** 2))
    second = (1 - 1 / gamma) * (T / press) * (G * mass * rho / rad ** 2)
    return -1 * np.min([first, second])

  
# epsilon values (PP, CNO)
def eps_pp(rho, T):
    return (1.07E-7) * (rho * 1E-5) * (X ** 2) * ((T * 1E-6) ** 4)


def eps_cno(rho, T):
    return (8.24E-26) * (rho * 1E-5) * (X) * (0.03 * X) * ((T * 1E-6) ** 19.9)


def kappa_func(T):
    kap_es = (0.02 * (1 + X))
    kap_ff = (1E24) * (Z + 0.0001) * ((rho * 1E-3) ** 0.7) * (T ** (-3.5))
    kap_H = (2.5E-32) * (Z / 0.02) * ((rho * 1E-3) ** 0.5) * (T ** 9)
    return 1 / (1 / kap_H + 1 / (np.max([kap_es, kap_ff])))


# function which solves stellar equations for a main sequence star of radius r_surf
def rksolver(r_surf):    
    
    # starting temperature and density at "centre" values
    T = T_c
    rho = rho_c
    
    # intial values for radius, epsilon, mass and luminosity
    r_0 = 0.01
    eps_0 = eps_pp(rho_c, T_c) + eps_cno(rho_c, T_c)
    M = (4 * math.pi / 3.0) * (r_0 ** 3) * rho_c
    lum = (4 * math.pi / 3.0) * (r_0 ** 3) * rho_c * eps_0
    
    # empty storage arrays for plotting purposes
    M_vals = []
    rho_vals = []
    T_vals = []
    L_vals = []    
    
    # RK step size, and surface radius to be looped to
    h = 10000
    
    for rad in np.linspace(0.01, r_surf, 100000):
        """This loops over a range of radius values ata  specified step size
            - kappa, pressure and epsilon values will be updated after each iteration of the loop
            - this loop implements fourth order Runge Kutta numerical integration for the 5 DEs that need to be solved
            - the new values of temp, density, luminosity, mass and opacity are added to after each iteration as we are
            integrating these values
        """
    
        # kappa function
        kappa = kappa_func(T, rho)
        # pressure calculation
        press = P(rho, T)
        # espilon calculation
        eps = eps_pp(rho, T) + eps_cno(rho, T)
    
        # temperature DE
        dTR = dT_dR(kappa, rho, rad, T, lum, M, press)
        dPT = dP_dT(rho, T)
        dPrho = dP_drho(rho, T)
    
        # runge kutta coefficients 1st Order
        l0 = h * dT_dR(kappa, rho, rad, T, lum, M, press)
        k0 = h * drho_dR(rho, rad, M, dPT, dTR, dPrho)
        m0 = h * dM_dR(rho, rad)
        n0 = h * dL_dR(rho, rad, eps)
        # p0 = h * dTau_dR(rho, kappa)
    
        # runge kutta coefficients 2nd Order
        l1 = h * dT_dR(kappa, rho + 0.5 * k0, rad + 0.5 * h, T + 0.5 * l0, lum + 0.5 * n0, M + 0.5 * m0, press)
        k1 = h * drho_dR(rho + 0.5 * k0, rad + 0.5 * h, M + 0.5 * m0, dPT, dTR, dPrho)
        m1 = h * dM_dR(rho + 0.5 * k0, rad + 0.5 * h)
        n1 = h * dL_dR(rho + 0.5 * k0, rad + 0.5 * h, eps)
        # p1 = h * dTau_dR(rho + 0.5 * k0, kappa)
    
        # runge kutta coefficients 3rd Order
        l2 = h * dT_dR(kappa, rho + 0.5 * k1, rad + 0.5 * h, T + 0.5 * l1, lum + 0.5 * n1, M + 0.5 * m1, press)
        k2 = h * drho_dR(rho + 0.5 * k1, rad + 0.5 * h, M + 0.5 * m1, dPT, dTR, dPrho)
        m2 = h * dM_dR(rho + 0.5 * k1, rad + 0.5 * h)
        n2 = h * dL_dR(rho + 0.5 * k1, rad + 0.5 * h, eps)
        # p2 = h * dTau_dR(rho + 0.5 * k1, kappa)
    
        # runge kutta coefficients 4th Order
        l3 = h * dT_dR(kappa, rho + k2, rad + h, T + l2, lum + n2, M + m2, press)
        k3 = h * drho_dR(rho + k2, rad + h, M + m2, dPT, dTR, dPrho)
        m3 = h * dM_dR(rho + k2, rad + h)
        n3 = h * dL_dR(rho + k2, rad + h, eps)
        # p3 = h * dTau_dR(rho + 0.5 * k2, kappa)
    
        # new temperature
        T += (1 / 6.0) * (l0 + 2 * l1 + 2 * l2 + l3)
        # new denisty
        rho += (1 / 6.0) * (k0 + 2 * k1 + 2 * k2 + k3)
        # new mass
        M += (1 / 6.0) * (m0 + 2 * m1 + 2 * m2 + m3)
        # new luminosity
        lum += (1 / 6.0) * (n0 + 2 * n1 + 2 * n2 + n3)
        # new tau
        # tau += (1/6.0)*(p0+2*p1+2*p2+p3)
    
        M_vals.append(M)
        rho_vals.append(rho)
        T_vals.append(T)
        L_vals.append(lum)
        
    # print out final values fo each parameter for reference
    print("Temperature: {}".format(T))
    print("Density: {}".format(rho))
    print("Mass: {}".format(M))
    print("Luminosity: {}".format(lum))
    
    return (r_surf, M_vals, rho_vals, T_vals, L_vals);

# scales data arrays to M/M_sun, rho/rho_c, T/T_c, L/L_sun
def scaleplots(M_vals, rho_vals, T_vals, L_vals):
    norm_M, norm_rho, norm_L, norm_T = [], [], [], []
    for m, rho, T, L in zip(M_vals, rho_vals, T_vals, L_vals):
        norm_M.append(m/m_sun)
        norm_rho.append(rho/rho_c)
        norm_T.append(T/T_c)
        norm_L.append(L/l_sun)
    return (norm_M, norm_rho, norm_L, norm_T)

"""example: for sun-like star
   I have commented out all this so that it does not plot all the graphs for
   every single generated star in generatesequence.py
   Uncomment to get individual star plots
"""

(r_surf, M_vals, rho_vals, T_vals, L_vals) = rksolver(0.75*r_sun);
(norm_M, norm_rho, norm_L, norm_T) = scaleplots(M_vals, rho_vals, T_vals, L_vals);

##applies the plot style defined in seabornstyle
#seabornstyle.set_style()

#plotting mass, density, temperature and luminosity as functions of r
fig = plt.figure()
plt.plot(np.linspace(0.01, r_surf, 100000), norm_M)
plt.title("Mass vs. Radius")
plt.xlabel("Radius")
plt.ylabel("Mass")
fig.show()

#fig2 = plt.figure()
#plt.plot(np.linspace(0.01, r_surf, 100000), norm_rho)
#plt.title("Density vs. Radius")
#plt.xlabel("Radius")
#plt.ylabel("Density")
## fig2.show()

#fig3 = plt.figure()
#plt.plot(np.linspace(0.01, r_surf, 100000), T_vals)
#plt.title("Temperature vs. Radius")
#plt.xlabel("Radius")
#plt.ylabel("Temperature")
## fig3.show()

#fig4 = plt.figure()
#plt.plot(np.linspace(0.01, r_surf, 100000), norm_L)
#plt.title("Luminosity vs. Radius")
#plt.xlabel("Radius")
#plt.ylabel("Luminosity")
## fig4.show()


# function calculating delta tau
def delta_tau(rad, rho, kappa, mass):
    return (kappa * rho ** 2) / abs(drho_dR(rad, rho, mass))


# surface condition that delta tau cannot go below 2/3
def surface_condition(rad, rho, kappa, mass):
    delta_limit = 2 / (3.0)
    delta_t = delta_tau(rad, rho, kappa, mass)
    if delta_t < delta_limit:
        return True
    return False


M_vals = []
rho_vals = []
T_vals = []
L_vals = []
radii = []
depths = []

T = T_c
rho = rho_c

r_0 = 0.01
eps_0 = eps_pp(rho_c, T_c) + eps_cno(rho_c, T_c)
M_0 = (4 * math.pi / 3.0) * (r_0 ** 3) * rho_c
lum_0 = (4 * math.pi / 3.0) * (r_0 ** 3) * rho_c * eps_0

# initial values and step size
M = M_0
lum = lum_0
rad = 0.01
h = 100000

for i in range(100000):

    # these will update with the 'new' values each time we loop through
    # kappa function
    kappa = kappa_func(T)
    # pressure calculation
    press = P(rho, T)
    # espilon calculation
    eps = eps_pp(rho, T) + eps_cno(rho, T)
    # tau calculation
    tau = kappa * rho * rad

    # temperature DE
    dTR = dT_dR(kappa, rho, rad, T, lum, M, press)
    dPT = dP_dT(rho, T)
    dPrho = dP_drho(rho, T)

    # runge kutta coefficients 1st Order
    l0 = h * dT_dR(kappa, rho, rad, T, lum, M, press)
    k0 = h * drho_dR(rho, rad, M)
    m0 = h * dM_dR(rho, rad)
    n0 = h * dL_dR(rho, rad, eps)
    p0 = h * dTau_dR(rho, kappa)

    # runge kutta coefficients 2nd Order
    l1 = h * dT_dR(kappa, rho + 0.5 * k0, rad + 0.5 * h, T + 0.5 * l0, lum + 0.5 * n0, M + 0.5 * m0, press)
    k1 = h * drho_dR(rho + 0.5 * k0, rad + 0.5 * h, M + 0.5 * m0)
    m1 = h * dM_dR(rho + 0.5 * k0, rad + 0.5 * h)
    n1 = h * dL_dR(rho + 0.5 * k0, rad + 0.5 * h, eps)
    p1 = h * dTau_dR(rho + 0.5 * k0, kappa)

    # runge kutta coefficients 3rd Order
    l2 = h * dT_dR(kappa, rho + 0.5 * k1, rad + 0.5 * h, T + 0.5 * l1, lum + 0.5 * n1, M + 0.5 * m1, press)
    k2 = h * drho_dR(rho + 0.5 * k1, rad + 0.5 * h, M + 0.5 * m1)
    m2 = h * dM_dR(rho + 0.5 * k1, rad + 0.5 * h)
    n2 = h * dL_dR(rho + 0.5 * k1, rad + 0.5 * h, eps)
    p2 = h * dTau_dR(rho + 0.5 * k1, kappa)

    # runge kutta coefficients 4th Order
    l3 = h * dT_dR(kappa, rho + k2, rad + h, T + l2, lum + n2, M + m2, press)
    k3 = h * drho_dR(rho + k2, rad + h, M + m2)
    m3 = h * dM_dR(rho + k2, rad + h)
    n3 = h * dL_dR(rho + k2, rad + h, eps)
    p3 = h * dTau_dR(rho + k2, kappa)
    # new temperature
    T = T + (1 / 6.0) * (l0 + 2 * l1 + 2 * l2 + l3)
    # new denisty
    rho = rho + (1 / 6.0) * (k0 + 2 * k1 + 2 * k2 + k3)
    # new mass
    M = M + (1 / 6.0) * (m0 + 2 * m1 + 2 * m2 + m3)
    # new luminosity
    lum = lum + (1 / 6.0) * (n0 + 2 * n1 + 2 * n2 + n3)
    # new optical depth
    tau = tau + (1 / 6.0) * (p0 + 2 * p1 + 2 * p2 + p3)

    # calculate delta tau and check surface condition
    del_tau = delta_tau(rad, rho, kappa, M)
    if surface_condition(rad, rho, kappa, M) == True:
        break

    M_vals.append(M)
    rho_vals.append(rho)
    T_vals.append(T)
    L_vals.append(lum)
    depths.append(tau)

    # adding step size to radius
    rad = rad + h
    radii.append(rad)

    # adaptive step sizes
    if rho_vals[-1] / rho_vals[0] < 0.5:
        h = 10000
    elif rho_vals[-1] / rho_vals[0] < 0.2:
        h = 5000
    elif rho_vals[-1] / rho_vals[0] < 0.05:
        h = 100

print("Temperature: {}".format(T))
print("Density: {}".format(rho))
print("Mass: {}".format(M))
print("Luminosity: {}".format(lum))

#scaling for plotting purposes
scaled_r = [r/rad for r in radii]
scaled_m = [m/M for m in M_vals]
scaled_l = [l/lum for l in L_vals]
scaled_rho = [p/rho_c for p in rho_vals]
scaled_t = [t/T_c for t in T_vals]

# plotting all curves on the same graph
fig = plt.figure()
plt.plot(scaled_r, scaled_m, label="Mass")
plt.plot(scaled_r, scaled_rho, label="Density")
plt.plot(scaled_r, scaled_t, label="Temperature")
plt.plot(scaled_r, scaled_l, label="Luminosity")
# plt.plot(scaled_r, depths)
plt.legend(loc="best")
plt.ylabel('Stuff')
plt.show()
