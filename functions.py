import math
import numpy as np
from astropy import constants as ast_const

# defining constants
h_bar = ast_const.hbar
G = ast_const.G
c = ast_const.c
m_e = ast_const.m_e
m_p = ast_const.m_p
k_b = ast_const.k_B
sigma_SB = ast_const.sigma_sb
gamma = 5/3
mew_mp = (1/((0.7)*2 + 0.75*(0.28) + 0.5*(0.02)))*m_p
a = 4*sigma_SB / c


# units rho_c: [kg/m**3]
rho_c = 56.55*1000
#units T_c: [K]
T_c = 3056


# epsilon values (PP, CNO)
def eps_pp(rho, T):
    return (1.07E-7)*(rho*1E-5)*(0.7**2)*((T*1E-6)**4)


def eps_cno(rho, T):
    return (8.24E-26)*(rho*1E-5)*(0.7)*(0.03*0.7)*((T*1E-6)**19.9)

eps = eps_pp(rho, T) + eps_cno(rho, T)


M_0 = (4*math.pi / 3)*(r_0**3)*rho_c
lum = (4*math.pi / 3)*(r_0**3)*rho_c*eps


# function returning Pressure
def P(rho, T):
    first_term = ((3*math.pi**2)**(2/3) / 5)*((h_bar**2)/m_e)*(rho/m_p)**(5/3)
    second_term = rho*(k_b*T)/mew_mp
    third_term = (1/3)*a*T**4
    return first_term + second_term + third_term


# DE for pressure in terms of rho
def dP_drho(rho, T):
    first = (((3*math.pi**2)**(2/3))/3)*((h_bar**2) / (m_e*m_p))*(rho/m_p)**(2/3)
    second = k_b*T / mew_mp
    return first + second


# DE for pressure in terms of temperature
def dP_dT(rho, T):
    return rho*k_b/(mew_mp) + (4/3.0)*a*(T**4)

# DE for mass in terms of radius
def dM_dR(rho, rad):
    return (4*math.pi*rad**2)*rho


# DE for luminosity in terms of radius
def dL_dR(rho, rad, eps):
    return (4*math.pi*rad**2)*rho*eps


# DE for tau in terms of radius
def dTau_dR(kappa, rho):
    return kappa*rho


# DE for density in terms of radius
def drho_dR(rho, rad, mass):
    num = -((G*mass*rho/(rad**2)) + dPT*dTR)
    denom = dPrho
    return num / denom


# DE for temperture in terms of radius
def dT_dR(kappa, rho, rad, T, lum, mass, press):
    first = 3*kappa*rho*lum / (16*math.pi*a*c*(T**3)*(rad**2))
    second = (1 - 1/gamma)*(T / press)*(G*mass*rho / rad**2)
    return np.min([first, second])
