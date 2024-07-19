import numpy as np
from context import PM
import matplotlib.pyplot as plt
from time import time

## Bragg mirror with increasing number of layers
mat1 = 1.5
mat2 = 1


unit = "um"
wav = 0.600

ep1 =  0.1
ep2 =  0.2

layers = np.arange(5, 181, 5)


print("Normal incidence, Bragg Mirror")
incidence = 0
materials = [1, mat1**2, mat2**2]

rs_s_te_um = []
ts_s_te_um = []

rs_t_te_um = []
ts_t_te_um = []

rs_a_te_um = []
ts_a_te_um = []

rs_i_te_um = []

rs_dn_te_um = []
ts_dn_te_um = []

rs_s_tm_um = []
ts_s_tm_um = []

rs_t_tm_um = []
ts_t_tm_um = []

rs_a_tm_um = []
ts_a_tm_um = []

rs_i_tm_um = []

rs_dn_tm_um = []
ts_dn_tm_um = []

for nb_couches in layers:

    ## Case 1: single layer, TE
    #structure = np.random.random(nb_couches*2+1)*w_mean
    structure = np.array([ep1, ep2]*nb_couches + [ep1])

    stack = [0]+[1,2]*nb_couches+[1,0]


    epaisseurs = np.concatenate(([0],structure,[0]))
    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,0)
    b = time()
    rs_s_te_um.append(R)
    ts_s_te_um.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te_um.append(R_ab)
    ts_a_te_um.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te_um.append(R_t)
    ts_t_te_um.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te_um.append(R_dn)
    ts_dn_te_um.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te_um.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm_um.append(R)
    ts_s_tm_um.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm_um.append(R_ab)
    ts_a_tm_um.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm_um.append(R_t)
    ts_t_tm_um.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm_um.append(R_dn)
    ts_dn_tm_um.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm_um.append(R_i)

rs_a_te_um = np.array(rs_a_te_um)
rs_a_tm_um = np.array(rs_a_tm_um)
ts_a_te_um = np.array(ts_a_te_um)
ts_a_tm_um = np.array(ts_a_tm_um)

rs_s_te_um = np.array(rs_s_te_um)
rs_s_tm_um = np.array(rs_s_tm_um)
ts_s_te_um = np.array(ts_s_te_um)
ts_s_tm_um = np.array(ts_s_tm_um)

rs_t_te_um = np.array(rs_t_te_um)
rs_t_tm_um = np.array(rs_t_tm_um)
ts_t_te_um = np.array(ts_t_te_um)
ts_t_tm_um = np.array(ts_t_tm_um)

rs_dn_te_um = np.array(rs_dn_te_um)
rs_dn_tm_um = np.array(rs_dn_tm_um)
ts_dn_te_um = np.array(ts_dn_te_um)
ts_dn_tm_um = np.array(ts_dn_tm_um)

rs_i_te_um = np.array(rs_i_te_um)
rs_i_tm_um = np.array(rs_i_tm_um)



unit = "nm"
wav = 600

ep1 =  100
ep2 =  200

layers = np.arange(5, 181, 5)


print("Normal incidence, Bragg Mirror")
incidence = 0
materials = [1, mat1**2, mat2**2]

rs_s_te_nm = []
ts_s_te_nm = []

rs_t_te_nm = []
ts_t_te_nm = []

rs_a_te_nm = []
ts_a_te_nm = []

rs_i_te_nm = []

rs_dn_te_nm = []
ts_dn_te_nm = []

rs_s_tm_nm = []
ts_s_tm_nm = []

rs_t_tm_nm = []
ts_t_tm_nm = []

rs_a_tm_nm = []
ts_a_tm_nm = []

rs_i_tm_nm = []

rs_dn_tm_nm = []
ts_dn_tm_nm = []

for nb_couches in layers:

    ## Case 1: single layer, TE
    #structure = np.random.random(nb_couches*2+1)*w_mean
    structure = np.array([ep1, ep2]*nb_couches + [ep1])

    stack = [0]+[1,2]*nb_couches+[1,0]


    epaisseurs = np.concatenate(([0],structure,[0]))
    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,0)
    b = time()
    rs_s_te_nm.append(R)
    ts_s_te_nm.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te_nm.append(R_ab)
    ts_a_te_nm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te_nm.append(R_t)
    ts_t_te_nm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te_nm.append(R_dn)
    ts_dn_te_nm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te_nm.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm_nm.append(R)
    ts_s_tm_nm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm_nm.append(R_ab)
    ts_a_tm_nm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm_nm.append(R_t)
    ts_t_tm_nm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm_nm.append(R_dn)
    ts_dn_tm_nm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm_nm.append(R_i)

rs_a_te_nm = np.array(rs_a_te_nm)
rs_a_tm_nm = np.array(rs_a_tm_nm)
ts_a_te_nm = np.array(ts_a_te_nm)
ts_a_tm_nm = np.array(ts_a_tm_nm)

rs_s_te_nm = np.array(rs_s_te_nm)
rs_s_tm_nm = np.array(rs_s_tm_nm)
ts_s_te_nm = np.array(ts_s_te_nm)
ts_s_tm_nm = np.array(ts_s_tm_nm)

rs_t_te_nm = np.array(rs_t_te_nm)
rs_t_tm_nm = np.array(rs_t_tm_nm)
ts_t_te_nm = np.array(ts_t_te_nm)
ts_t_tm_nm = np.array(ts_t_tm_nm)

rs_dn_te_nm = np.array(rs_dn_te_nm)
rs_dn_tm_nm = np.array(rs_dn_tm_nm)
ts_dn_te_nm = np.array(ts_dn_te_nm)
ts_dn_tm_nm = np.array(ts_dn_tm_nm)

rs_i_te_nm = np.array(rs_i_te_nm)
rs_i_tm_nm = np.array(rs_i_tm_nm)


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
axs[0,0].plot(layers, np.abs(rs_s_te_um-rs_s_te_nm), 'b-v', label="abeles", markersize=4)
axs[0,0].plot(layers, np.abs(rs_dn_te_um-rs_dn_te_nm), 'r-o', label="D2N", markersize=4)
axs[0,0].plot(layers, np.abs(rs_t_te_um-rs_t_te_nm), 'g-^', label="T", markersize=4)
axs[0,0].plot(layers, np.abs(rs_i_te_um-rs_i_te_nm), 'c-+', label="Impedance", markersize=4)
axs[0,0].set_ylabel("Reflection relative error TE Normal incidence")
axs[0,0].set_xlabel("Nb Layers")
##axs[0,0].set_ylim([0-.0001,.15])
#axs[0,0].set_yscale("log")
axs[0,0].legend()


axs[0,1].plot(layers, np.abs(rs_s_tm_um-rs_s_tm_nm), 'b-v', label="abeles", markersize=4)
axs[0,1].plot(layers, np.abs(rs_dn_tm_um-rs_dn_tm_nm), 'r-o', label="D2N", markersize=4)
axs[0,1].plot(layers, np.abs(rs_t_tm_um-rs_t_tm_nm), 'g-^', label="T", markersize=4)
axs[0,1].plot(layers, np.abs(rs_i_tm_um-rs_i_tm_nm), 'c-+', label="Impedance", markersize=4)
axs[0,1].set_ylabel("Reflection relative error TM Normal incidence")
axs[0,1].set_xlabel("Nb Layers")
##axs[0,1].set_ylim([-0.001,.15])
#axs[0,1].set_yscale("log")
axs[0,1].legend()


axs[1,0].plot(layers, np.abs(ts_s_te_um-ts_s_te_nm), 'b-v', label="abeles", markersize=4)
axs[1,0].plot(layers, np.abs(ts_dn_te_um-ts_dn_te_nm), 'r-o', label="D2N", markersize=4)
axs[1,0].plot(layers, np.abs(ts_t_te_um-ts_t_te_nm), 'g-^', label="T", markersize=4)
axs[1,0].set_ylabel("Transmission relative error TE Normal incidence")
axs[1,0].set_xlabel("Nb Layers")
#axs[1,0].set_ylim([-0.001,.15])
#axs[1,0].set_yscale("log")
axs[1,0].legend()


axs[1,1].plot(layers, np.abs(ts_s_tm_um-ts_s_tm_nm), 'b-v', label="abeles", markersize=4)
axs[1,1].plot(layers, np.abs(ts_dn_tm_um-ts_dn_tm_nm), 'r-o', label="D2N", markersize=4)
axs[1,1].plot(layers, np.abs(ts_t_tm_um-ts_t_tm_nm), 'g-^', label="T", markersize=4)
axs[1,1].set_ylabel("Transmission relative error TM Normal incidence")
axs[1,1].set_xlabel("Nb Layers")
#axs[1,1].set_ylim([-0.001,.15])
#axs[1,1].set_yscale("log")
axs[1,1].legend()
plt.tight_layout()
plt.show()
