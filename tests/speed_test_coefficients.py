import numpy as np
from context import PM
import matplotlib.pyplot as plt
from time import time

## Computation times

nb_iter = 500 #averaging
layers = np.arange(5, 181, 5)

wav = 3.5
ep1 = 2
ep2 = 3
materials = [1, 1.5**2, 2**2]
incidence = 15 * np.pi/180

times_s_tm = np.zeros(len(layers), dtype=float)

times_t_tm = np.zeros(len(layers), dtype=float)

times_a_tm = np.zeros(len(layers), dtype=float)

times_i_tm = np.zeros(len(layers), dtype=float)

times_dn_tm = np.zeros(len(layers), dtype=float)

for i in range(nb_iter):
    for j, nb_couches in enumerate(layers):

        ## Case 1: single layer, TE
        #structure = np.random.random(nb_couches*2+1)*w_mean
        structure = np.array([ep1, ep2]*nb_couches + [ep1])

        stack = [0]+[1,2]*nb_couches+[1,0]


        epaisseurs = np.concatenate(([0],structure,[0]))


        chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
        a = time()
        r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)
        b = time()
        times_s_tm[j] += (b-a)/nb_iter



        chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
        a = time()
        r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)
        b = time()
        times_a_tm[j] += (b-a)/nb_iter


        chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
        a = time()
        r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)
        b = time()
        times_t_tm[j] += (b-a)/nb_iter


        chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
        a = time()
        r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)
        b = time()
        times_dn_tm[j] += (b-a)/nb_iter


        chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
        a = time()
        r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)
        b = time()
        times_i_tm[j] += (b-a)/nb_iter


plt.figure(figsize=(7,7))
plt.plot(layers, times_a_tm, 'o', label="abeles")
plt.plot(layers, times_dn_tm, 'o', label="D2N")
plt.plot(layers, times_s_tm, 'o', label="S")
plt.plot(layers, times_t_tm, 'o', label="T")
plt.plot(layers, times_i_tm, 'o', label="Impedance")
plt.title("Computation times")
plt.xlabel("Nb Layers")
plt.legend()
plt.show()
