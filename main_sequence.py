import math
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
# matplotlib.use('TkAgg')  # needed for Yilda to make matplotlib work in virtualenv
from RK_solver import star_gen
import csv
import os

root = os.path.dirname(os.path.abspath(__file__))

def mainsequence(T):

    print("=== NEW STAR GENERATION === Tc = {}".format(T))

    """
        values being pulled from the star_gen function
        star_gen returns (in dictionary):
        arrays: raddi, M_vals, T_vals, rho_vals, L_vals
        values: norm_R, norm_M, norm_L, norm_rho, norm_T
    """
    star = star_gen(T)
    radii = star["radii"]
    M_vals = star["M_vals"]
    T_vals = star["T_vals"]
    rho_vals = star["rho_vals"]
    L_vals = star["L_vals"]

    norm_R = star["norm_R"]
    norm_M = star["norm_M"]
    norm_L = star["norm_L"]
    norm_rho = star["norm_rho"]
    norm_T = star["norm_T"]

    # plot the various outputs for the star and store in figures_init
    plt.figure()
    plt.plot(norm_R, norm_M, label=("Mass" + str(M_vals[-1])))
    plt.plot(norm_R, norm_rho, label="Density")
    plt.plot(norm_R, norm_T, label="Temperature")
    plt.plot(norm_R, norm_L, label="Luminosity")
    # plt.plot(norm_R, depths)
    plt.legend(loc="best")
    plt.xlabel("Radius {}".format(str(radii[-1])))
    plt.ylabel('M/M* Rho/Rho* T/Tc L/L*')
    plt.savefig("{}/figures_init/test_star_{}.png".format(root, T))
    # plt.show()

    # printing the final values after rho_c is found
    print('FINAL VALUES')
    print("=============================")
    print("Radius: {}".format(str(radii[-1])))
    print("Mass: {}".format(str(M_vals[-1])))
    print("Temperature: {}".format(str(T_vals[-1])))
    print("Density: {}".format(str(rho_vals[-1])))
    print("Luminosity: {}".format(str(L_vals[-1])))

    # csv file for writing star data
    data_file = "{}/star_data.csv".format(root)
    with open(data_file, 'a') as write_f:
        writer = csv.writer(write_f, delimiter=',')
        values_all = [T, rho_vals[0], radii[-1], M_vals[-1], T_vals[-1], rho_vals[-1], L_vals[-1]]
        writer.writerow(values_all)
        # writer.writerow('\n')
    write_f.close()

# =============== MULTIPROCESSING =================================
temp_range = [1.6E6, 1.7E6]
# if __name__ == '__main__':
#     jobs = []
#     for t in temp_range:
#         p = multiprocessing.Process(target=mainsequence, args=(t,))
#         jobs.append(p)
#         p.start()

for temp in temp_range:
    mainsequence(temp)
