import math
import numpy as np
from astropy import constants as ast_const
import matplotlib.pyplot as plt
import time
import csv
import os


root = os.path.dirname(os.path.abspath(__file__))


def star_gen(T):
    # defining constants
    h_bar = ast_const.hbar.value
    G = ast_const.G.value
    c = ast_const.c.value
    m_e = ast_const.m_e.value
    m_p = ast_const.m_p.value
    k_b = ast_const.k_B.value
    r_sun = ast_const.R_sun.value
    sigma_SB = ast_const.sigma_sb.value
    gamma = 5 / 3.0
    a = 4 * sigma_SB / c

    # mass fractions
    X = 0.7
    Y = 0.28
    Z = 0.02
    mew_mp = (1 / ((X) * 2 + 0.75 * (Y) + 0.5 * (Z))) * m_p

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
    def dTau_dR(rho, kappa):
        return kappa * rho

    # DE for density in terms of radius
    def drho_dR(rho, rad, mass, dTR, dPT, dPrho):
        num = -((G * mass * rho / (rad ** 2)) + dPT * dTR)
        denom = dPrho
        return num / denom

    # DE for temperture in terms of radius
    def dT_dR(kappa, rho, rad, T, lum, mass, press):
        first = 3 * kappa * rho * lum / (16 * math.pi * a * c * (T ** 3) * (rad ** 2))
        second = (1 - 1 / gamma) * (T / press) * (G * mass * rho / rad ** 2)
        return -1 * np.min([first, second])

    # units T_c: [K]
    T_c = T

    # rho_c = 5.8560E4

    # epsilon values (PP, CNO)
    def eps_pp(rho, T):
        return (1.07E-7) * (rho * 1E-5) * (X ** 2) * ((T * 1E-6) ** 4)

    def eps_cno(rho, T):
        return (8.24E-26) * (rho * 1E-5) * (X) * (0.03 * X) * ((T * 1E-6) ** 19.9)

    def kappa_func(rho, T):
        kap_es = (0.02 * (1 + X))
        kap_ff = (1E24) * (Z + 0.0001) * ((rho * 1E-3) ** 0.7) * (T ** (-3.5))
        kap_H = (2.5E-32) * (Z / 0.02) * ((rho * 1E-3) ** 0.5) * (T ** 9)
        return 1 / (1 / kap_H + 1 / (np.max([kap_es, kap_ff])))

    # surface condition that delta tau cannot go below 2/3
    # def surface_condition(rad, rho, kappa, mass, dTR, dPT, dPrho):
    #     delta_t = delta_tau(rad, rho, kappa, mass, dTR, dPT, dPrho)
    #     if delta_t < 2 / 3:
    #         return True
    #     return False

    # function to calculate a trail rho_c solution
    def trial(lum, rad, temp):
        num = lum - 4 * math.pi * sigma_SB * (rad ** 2) * (temp ** 4)
        denom = math.sqrt(4 * math.pi * sigma_SB * (rad ** 2) * (temp ** 4) * lum)
        return num / denom


    # function which solves stellar equations for a main sequence star of radius r_surf
    def rksolver(rho_c):
        M_vals = []
        rho_vals = []
        T_vals = []
        L_vals = []
        radii = []
        depths = []

        T = T_c
        rho = rho_c

        r_0 = 0.001
        eps_0 = eps_pp(rho, T) + eps_cno(rho, T)
        M_0 = (4 * math.pi / 3.0) * (r_0 ** 3) * rho
        lum_0 = (4 * math.pi / 3.0) * (r_0 ** 3) * rho * eps_0

        # initial values and step size
        M = M_0
        lum = lum_0
        rad = r_0
        tau = kappa_func(rho, T) * rho * rad
        h = 100000

        count = 0
        #TODO: implement more meaningful loop
        # continuous loop, number of iterations arbitrarily large
        for i in range(100000000):
            count += 1

            # kappa, pressure and epsilon calculations
            kappa = kappa_func(rho, T)
            press = P(rho, T)
            eps = eps_pp(rho, T) + eps_cno(rho, T)

            # equations for single differentials
            dTR = dT_dR(kappa, rho, rad, T, lum, M, press)
            dPT = dP_dT(rho, T)
            dPrho = dP_drho(rho, T)

            # append values to arrays for storage and plotting
            M_vals.append(M)
            rho_vals.append(rho)
            T_vals.append(T)
            L_vals.append(lum)
            depths.append(tau)
            radii.append(rad)

            # runge kutta coefficients 1st Order
            l0 = h * dT_dR(kappa, rho, rad, T, lum, M, press)
            k0 = h * drho_dR(rho, rad, M, dTR, dPT, dPrho)
            m0 = h * dM_dR(rho, rad)
            n0 = h * dL_dR(rho, rad, eps)
            p0 = h * dTau_dR(rho, kappa)

            # kappa, pressure and epsilon calculations (1st order)
            kappa = kappa_func(rho + 0.5 * k0, T + 0.5 * l0)
            press = P(rho + 0.5 * k0, T + 0.5 * l0)
            eps = eps_pp(rho + 0.5 * k0, T + 0.5 * l0) + eps_cno(rho + 0.5 * k0, T + 0.5 * l0)

            # single differentials (1st order)
            dTR = dT_dR(kappa, rho + 0.5 * k0, rad + 0.5 * h, T + 0.5 * l0, lum + 0.5 * n0, M + 0.5 * m0, press)
            dPT = dP_dT(rho + 0.5 * k0, T + 0.5 * l0)
            dPrho = dP_drho(rho + 0.5 * k0, T + 0.5 * l0)

            # runge kutta coefficients second order
            l1 = h * dT_dR(kappa, rho + 0.5 * k0, rad + 0.5 * h, T + 0.5 * l0, lum + 0.5 * n0, M + 0.5 * m0, press)
            k1 = h * drho_dR(rho + 0.5 * k0, rad + 0.5 * h, M + 0.5 * m0, dTR, dPT, dPrho)
            m1 = h * dM_dR(rho + 0.5 * k0, rad + 0.5 * h)
            n1 = h * dL_dR(rho + 0.5 * k0, rad + 0.5 * h, eps)
            p1 = h * dTau_dR(rho + 0.5 * k0, kappa)

            # kappa, pressure and epsilon calculations (2nd order)
            kappa = kappa_func(rho + 0.5 * k1, T + 0.5 * l1)
            press = P(rho + 0.5 * k1, T + 0.5 * l1)
            eps = eps_pp(rho + 0.5 * k1, T + 0.5 * l1) + eps_cno(rho + 0.5 * k1, T + 0.5 * l1)

            # single differentials (2nd order)
            dTR = dT_dR(kappa, rho + 0.5 * k1, rad + 0.5 * h, T + 0.5 * l1, lum + 0.5 * n1, M + 0.5 * m1, press)
            dPT = dP_dT(rho + 0.5 * k1, T + 0.5 * l1)
            dPrho = dP_drho(rho + 0.5 * k1, T + 0.5 * l1)

            # runge kutta coefficients 3rd Order
            l2 = h * dT_dR(kappa, rho + 0.5 * k1, rad + 0.5 * h, T + 0.5 * l1, lum + 0.5 * n1, M + 0.5 * m1, press)
            k2 = h * drho_dR(rho + 0.5 * k1, rad + 0.5 * h, M + 0.5 * m1, dTR, dPT, dPrho)
            m2 = h * dM_dR(rho + 0.5 * k1, rad + 0.5 * h)
            n2 = h * dL_dR(rho + 0.5 * k1, rad + 0.5 * h, eps)
            p2 = h * dTau_dR(rho + 0.5 * k1, kappa)

            # kappa, pressure and epsilon calculations (3rd order)
            kappa = kappa_func(rho + k2, T + l2)
            press = P(rho + k2, T + l2)
            eps = eps_pp(rho + k2, T + l2) + eps_cno(rho + k2, T + l2)

            # single differentials (3rd order)
            dTR = dT_dR(kappa, rho + k1, rad + h, T + l2, lum + n2, M + m2, press)
            dPT = dP_dT(rho + k2, T + l2)
            dPrho = dP_drho(rho + k2, T + l2)

            # 4th order Runge Kutta coefficients
            l3 = h * dT_dR(kappa, rho + k2, rad + h, T + l2, lum + n2, M + m2, press)
            k3 = h * drho_dR(rho + k2, rad + h, M + m2, dTR, dPT, dPrho)
            m3 = h * dM_dR(rho + k2, rad + h)
            n3 = h * dL_dR(rho + k2, rad + h, eps)
            p3 = h * dTau_dR(rho + k2, kappa)

            # kappa, pressure and epsilon calculations (4th order)
            kappa = kappa_func(rho + 0.5*k1, T + 0.5*l1)
            press = P(rho + 0.5 * k1, T + 0.5 * l1)
            eps = eps_pp(rho + 0.5 * k1, T + 0.5 * l1) + eps_cno(rho + 0.5 * k1, T + 0.5 * l1)

            # single differentials (4th order)
            dTR = dT_dR(kappa, rho + 0.5 * k1, rad + 0.5 * h, T + 0.5 * l1, lum + 0.5 * n1, M + 0.5 * m1, press)
            dPT = dP_dT(rho + 0.5 * k1, T + 0.5 * l1)
            dPrho = dP_drho(rho + 0.5 * k1, T + 0.5 * l1)

            # updated temp, density, mass, luminosity and optical depth
            T = T + (1 / 6.0) * (l0 + 2 * l1 + 2 * l2 + l3)
            rho = rho + (1 / 6.0) * (k0 + 2 * k1 + 2 * k2 + k3)
            M = M + (1 / 6.0) * (m0 + 2 * m1 + 2 * m2 + m3)
            lum = lum + (1 / 6.0) * (n0 + 2 * n1 + 2 * n2 + n3)
            tau = tau + (1 / 6.0) * (p0 + 2 * p1 + 2 * p2 + p3)

            # calculate delta tau and check surface condition
            dRho_0 = (1 / 6.0) * (k0 + 2 * k1 + 2 * k2 + k3)
            new_dRho_r = dRho_0 / h
            del_tau = (kappa * rho ** 2) / abs(new_dRho_r)

            # if count % 10000 == 0:
            #     print('\n')
            #     print("R/Rsun:", rad / r_sun, "dTau:", del_tau, "Mass:", M)

            # check delta tau surface condition, mass condition and iteration condition
            # TODO: implement this in function or as part of loop
            if (del_tau < (2 / 3)) or (M > 1e32) or rad > 20*r_sun:
                # print("Mass: {}, Luminosity: {}, Radius: {}, Density: {}, dTau: {}"
                #       .format(str(M), str(lum), str(rad), str(rho), str(del_tau)))

                # evaluating the f function as surface condition reached, part of bisection
                f = trial(lum, rad, T)
                break

            # adding step size to radius
            rad = rad + h

            # adaptive step sizes
            if rho / rho_vals[0] < 0.05:
                h = 10000
            elif rho / rho_vals[0] < 0.005:
                h = 5000
            elif rho / rho_vals[0] < 0.0005:
                h = 1000
        # return dictionary with all necessary information
        return {"radii": radii, "M_vals": M_vals, "rho_vals": rho_vals, "T_vals": T_vals, "L_vals": L_vals,
                "depths": depths, "trial_f": f}

    def bisection(a):
        values = []
        # print("F1:", rksolver(a)["trial_f"])
        while rksolver(a)["trial_f"] > 0:
            values.append(a)
            a = a - 0.02 * a
        #     print('\n')
        #     print("Stepping down to a root, rho_c, F1:", a, rksolver(a)["trial_f"])
        # print(values)
        b = values[-1]

        # print('starting bisection with lower, upper', a, b)
        # print('starting F1', a)
        f1 = rksolver(a)["trial_f"]
        # print('starting F2', b)
        f2 = rksolver(b)["trial_f"]

        m = (a + b) / 2.0
        # print('Starting Fnew', m)
        f_new = rksolver(m)["trial_f"]
        # print('\n')
        # print('F1, F2, Fnew', f1, f2, f_new)
        # print('\n')

        while not abs(f_new) < 0.5:
            if np.sign(f_new) + np.sign(f1) == 0:

                a = a
                f1 = f1
                b = m
                f2 = f_new
                # print('\n')
                # print('F1, F2, Fnew', f1, f2, f_new)
                # print('\n')
            elif np.sign(f_new) + np.sign(f2) == 0:
                a = m
                f1 = f_new
                b = b
                f2 = f2
                # print('\n')
                # print('F1, F2, Fnew', f1, f2, f_new)
                # print('\n')
            m = (a + b) / 2.0
            # this condition ensure we can converge
            if abs(a - b) < 0.000000001:
                m = b
                return (rksolver(b))
            # print('\n')
            # print("Starting Fnew, low, mid, upper", a, m, b)
            f_new = rksolver(m)["trial_f"]
            # print('\n')
            # print('F1, F2, Fnew', f1, f2, f_new)
            # print('\n')
        return rksolver(m)

    # testing rho_c with lower limit and upper limit for bisection
    # low_rho_c = 300
    high_rho_c = 500000

    # running the  bisection function over the runge kutta solver
    solved = bisection(high_rho_c)
    radii = solved["radii"]
    M_vals = solved["M_vals"]
    rho_vals = solved["rho_vals"]
    T_vals = solved["T_vals"]
    L_vals = solved["L_vals"]
    depths = solved["depths"]

    # scaling for plotting purposes
    norm_R = [r / radii[-1] for r in radii]
    norm_M = [m / M_vals[-1] for m in M_vals]
    norm_L = [l / L_vals[-1] for l in L_vals]
    norm_rho = [p / rho_vals[0] for p in rho_vals]
    norm_T = [t / T_c for t in T_vals]

    # printing the final values after rho_c is found
    print('FINAL VALUES')
    print("=============================")
    print("Radius: {}".format(str(radii[-1])))
    print("Mass: {}".format(str(M_vals[-1])))
    print("Temperature: {}".format(str(T_vals[-1])))
    print("Density: {}".format(str(rho_vals[-1])))
    print("Luminosity: {}".format(str(L_vals[-1])))

    # applies the plot style defined in seabornstyle
    # seabornstyle.set_style()

    """example: for sun-like star
       I have commented out all this so that it does not plot the graph for
       every single generated star in generatesequence.py
       Uncomment to get plot for each star
    """
    # plotting all curves on the same graph
    plt.figure()
    plt.plot(norm_R, norm_M, label=("Mass" + str(M_vals[-1])))
    plt.plot(norm_R, norm_rho, label="Density")
    plt.plot(norm_R, norm_T, label="Temperature")
    plt.plot(norm_R, norm_L, label="Luminosity")
    # plt.plot(norm_R, depths)
    plt.legend(loc="best")
    plt.xlabel("Radius {}".format(str(radii[-1])))
    plt.ylabel('M/M* Rho/Rho* T/Tc L/L*')
    plt.savefig("{}/figures_init/test_star_{}.png".format(root, T_c))
    # plt.show()

    # csv file for writing star data
    data_file = "{}/star_data.csv".format(root)
    with open(data_file, 'a') as write_f:
        writer = csv.writer(write_f, delimiter=',')
        values_all = [T_c, rho_vals[0], radii[-1], M_vals[-1], T_vals[-1], rho_vals[-1], L_vals[-1]]
        writer.writerow(values_all)
        # writer.writerow('\n')
    write_f.close()


# TODO: this is the interim generation function, it takes a Tc value and prints/plots results
temps = [12.5E6]
for temp in temps:
    star_gen(temp)
    print("WROTE: {}".format(temp))
