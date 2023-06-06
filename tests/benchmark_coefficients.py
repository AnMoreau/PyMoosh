import numpy as np
from context import PM
import matplotlib.pyplot as plt
from time import time

## Bragg mirror with increasing number of layers
mat1 = 1.5
mat2 = 1


unit = "um"
wav = .600

ep1 =  wav/(4*mat1)
ep2 =  wav/(4*mat2)

layers = np.arange(5, 181, 5)


print("Normal incidence, Bragg Mirror")
incidence = 0
materials = [1, mat1**2, mat2**2]

rs_s_te = []
ts_s_te = []

rs_t_te = []
ts_t_te = []

rs_a_te = []
ts_a_te = []

rs_i_te = []

rs_dn_te = []
ts_dn_te = []

rs_s_tm = []
ts_s_tm = []

rs_t_tm = []
ts_t_tm = []

rs_a_tm = []
ts_a_tm = []

rs_i_tm = []

rs_dn_tm = []
ts_dn_tm = []

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
    rs_s_te.append(R)
    ts_s_te.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm.append(R_i)

rs_a_te = np.array(rs_a_te)
rs_a_tm = np.array(rs_a_tm)
ts_a_te = np.array(ts_a_te)
ts_a_tm = np.array(ts_a_tm)

rs_s_te = np.array(rs_s_te)
rs_s_tm = np.array(rs_s_tm)
ts_s_te = np.array(ts_s_te)
ts_s_tm = np.array(ts_s_tm)

rs_t_te = np.array(rs_t_te)
rs_t_tm = np.array(rs_t_tm)
ts_t_te = np.array(ts_t_te)
ts_t_tm = np.array(ts_t_tm)

rs_dn_te = np.array(rs_dn_te)
rs_dn_tm = np.array(rs_dn_tm)
ts_dn_te = np.array(ts_dn_te)
ts_dn_tm = np.array(ts_dn_tm)

rs_i_te = np.array(rs_i_te)
rs_i_tm = np.array(rs_i_tm)


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
axs[0,0].set_ylabel("Reflection relative error TE Normal incidence")
axs[0,0].set_xlabel("Nb Layers")
#axs[0,0].set_ylim([0-.0001,.15])
#axs[0,0].set_yscale("log")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
axs[0,1].set_ylabel("Reflection relative error TM Normal incidence")
axs[0,1].set_xlabel("Nb Layers")
#axs[0,1].set_ylim([-0.001,.15])
#axs[0,1].set_yscale("log")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te)/np.abs(ts_s_te), 'b-v', label="abeles", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te)/np.abs(ts_s_te), 'r-o', label="D2N", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te)/np.abs(ts_s_te), 'g-^', label="T", markersize=4)
axs[1,0].set_ylabel("Transmission relative error TE Normal incidence")
axs[1,0].set_xlabel("Nb Layers")
axs[1,0].set_ylim([-0.001,.15])
#axs[1,0].set_yscale("log")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm)/np.abs(ts_s_tm), 'b-v', label="abeles", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm)/np.abs(ts_s_tm), 'r-o', label="D2N", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm)/np.abs(ts_s_tm), 'g-^', label="T", markersize=4)
axs[1,1].set_ylabel("Transmission relative error TM Normal incidence")
axs[1,1].set_xlabel("Nb Layers")
axs[1,1].set_ylim([-0.001,.15])
#axs[1,1].set_yscale("log")
axs[1,1].legend()
plt.tight_layout()
plt.show()



print("Large incidence:")
incidence = 60*np.pi/180
materials = [1, mat1**2, mat2**2]

rs_s_te = []
ts_s_te = []

rs_t_te = []
ts_t_te = []

rs_a_te = []
ts_a_te = []

rs_i_te = []

rs_dn_te = []
ts_dn_te = []

rs_s_tm = []
ts_s_tm = []

rs_t_tm = []
ts_t_tm = []

rs_a_tm = []
ts_a_tm = []

rs_i_tm = []

rs_dn_tm = []
ts_dn_tm = []

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
    rs_s_te.append(R)
    ts_s_te.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm.append(R_i)

rs_a_te = np.array(rs_a_te)
rs_a_tm = np.array(rs_a_tm)
ts_a_te = np.array(ts_a_te)
ts_a_tm = np.array(ts_a_tm)

rs_s_te = np.array(rs_s_te)
rs_s_tm = np.array(rs_s_tm)
ts_s_te = np.array(ts_s_te)
ts_s_tm = np.array(ts_s_tm)

rs_t_te = np.array(rs_t_te)
rs_t_tm = np.array(rs_t_tm)
ts_t_te = np.array(ts_t_te)
ts_t_tm = np.array(ts_t_tm)

rs_dn_te = np.array(rs_dn_te)
rs_dn_tm = np.array(rs_dn_tm)
ts_dn_te = np.array(ts_dn_te)
ts_dn_tm = np.array(ts_dn_tm)

rs_i_te = np.array(rs_i_te)
rs_i_tm = np.array(rs_i_tm)


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
axs[0,0].set_ylabel("Reflection relative error TE Large incidence")
axs[0,0].set_xlabel("Nb Layers")
#axs[0,0].set_ylim([-0.001,.15])
#axs[0,0].set_yscale("log")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
axs[0,1].set_ylabel("Reflection relative error TM Large incidence")
axs[0,1].set_xlabel("Nb Layers")
#axs[0,1].set_ylim([-0.001,.15])
#axs[0,1].set_yscale("log")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te)/np.abs(ts_s_te), 'b-v', label="abeles", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te)/np.abs(ts_s_te), 'r-o', label="D2N", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te)/np.abs(ts_s_te), 'g-^', label="T", markersize=4)
axs[1,0].set_ylabel("Transmission relative error TE Large incidence")
axs[1,0].set_xlabel("Nb Layers")
#axs[1,0].set_ylim([-0.001,.15])
#axs[1,0].set_yscale("log")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm)/np.abs(ts_s_tm), 'b-v', label="abeles", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm)/np.abs(ts_s_tm), 'r-o', label="D2N", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm)/np.abs(ts_s_tm), 'g-^', label="T", markersize=4)
axs[1,1].set_ylabel("Transmission relative error TM Large incidence")
axs[1,1].set_xlabel("Nb Layers")
#axs[1,1].set_ylim([-0.001,.15])
#axs[1,1].set_yscale("log")
axs[1,1].legend()
plt.tight_layout()
plt.show()



print("Intermediate incidence:")
incidence = 15*np.pi/180
materials = [1, mat1**2, mat2**2]

rs_s_te = []
ts_s_te = []

rs_t_te = []
ts_t_te = []

rs_a_te = []
ts_a_te = []

rs_i_te = []

rs_dn_te = []
ts_dn_te = []

rs_s_tm = []
ts_s_tm = []

rs_t_tm = []
ts_t_tm = []

rs_a_tm = []
ts_a_tm = []

rs_i_tm = []

rs_dn_tm = []
ts_dn_tm = []

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
    rs_s_te.append(R)
    ts_s_te.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm.append(R_i)

rs_a_te = np.array(rs_a_te)
rs_a_tm = np.array(rs_a_tm)
ts_a_te = np.array(ts_a_te)
ts_a_tm = np.array(ts_a_tm)

rs_s_te = np.array(rs_s_te)
rs_s_tm = np.array(rs_s_tm)
ts_s_te = np.array(ts_s_te)
ts_s_tm = np.array(ts_s_tm)

rs_t_te = np.array(rs_t_te)
rs_t_tm = np.array(rs_t_tm)
ts_t_te = np.array(ts_t_te)
ts_t_tm = np.array(ts_t_tm)

rs_dn_te = np.array(rs_dn_te)
rs_dn_tm = np.array(rs_dn_tm)
ts_dn_te = np.array(ts_dn_te)
ts_dn_tm = np.array(ts_dn_tm)

rs_i_te = np.array(rs_i_te)
rs_i_tm = np.array(rs_i_tm)


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
axs[0,0].set_ylabel("Reflection relative error TE Intermediate incidence")
axs[0,0].set_xlabel("Nb Layers")
#axs[0,0].set_ylim([-0.001,.15])
#axs[0,0].set_yscale("log")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
axs[0,1].set_ylabel("Reflection relative error TM Intermediate incidence")
axs[0,1].set_xlabel("Nb Layers")
#axs[0,1].set_ylim([-0.001,.15])
#axs[0,1].set_yscale("log")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te)/np.abs(ts_s_te), 'b-v', label="abeles", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te)/np.abs(ts_s_te), 'r-o', label="D2N", markersize=4)
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te)/np.abs(ts_s_te), 'g-^', label="T", markersize=4)
axs[1,0].set_ylabel("Transmission relative error TE Intermediate incidence")
axs[1,0].set_xlabel("Nb Layers")
axs[1,0].set_ylim([-0.001,.15])
#axs[1,0].set_yscale("log")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm)/np.abs(ts_s_tm), 'b-v', label="abeles", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm)/np.abs(ts_s_tm), 'r-o', label="D2N", markersize=4)
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm)/np.abs(ts_s_tm), 'g-^', label="T", markersize=4)
axs[1,1].set_ylabel("Transmission relative error TM Intermediate incidence")
axs[1,1].set_xlabel("Nb Layers")
axs[1,1].set_ylim([-0.001,.15])
#axs[1,1].set_yscale("log")
axs[1,1].legend()
plt.tight_layout()
plt.show()

## Frustrated total internal reflection


print("Frustrated total internal reflection:")
materials = [1, mat1**2, mat2**2]

incidence = np.arcsin(1/mat1)+0.3

stack = [1, 0, 1]

distances = wav * np.arange(0.01, 7, 0.05)
if (unit == "um"):
    distances = distances*1e-3

rs_s_te = []
ts_s_te = []

rs_t_te = []
ts_t_te = []

rs_a_te = []
ts_a_te = []

rs_i_te = []

rs_dn_te = []
ts_dn_te = []

rs_s_tm = []
ts_s_tm = []

rs_t_tm = []
ts_t_tm = []

rs_a_tm = []
ts_a_tm = []

rs_i_tm = []

rs_dn_tm = []
ts_dn_tm = []

for dist in distances:

    ## Case 1: single layer, TE

    epaisseurs = np.concatenate(([0],[dist],[0]))
    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm.append(R_i)

rs_a_te = np.array(rs_a_te)
rs_a_tm = np.array(rs_a_tm)
ts_a_te = np.array(ts_a_te)
ts_a_tm = np.array(ts_a_tm)

rs_s_te = np.array(rs_s_te)
rs_s_tm = np.array(rs_s_tm)
ts_s_te = np.array(ts_s_te)
ts_s_tm = np.array(ts_s_tm)

rs_t_te = np.array(rs_t_te)
rs_t_tm = np.array(rs_t_tm)
ts_t_te = np.array(ts_t_te)
ts_t_tm = np.array(ts_t_tm)

rs_dn_te = np.array(rs_dn_te)
rs_dn_tm = np.array(rs_dn_tm)
ts_dn_te = np.array(ts_dn_te)
ts_dn_tm = np.array(ts_dn_tm)

rs_i_te = np.array(rs_i_te)
rs_i_tm = np.array(rs_i_tm)


fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
axs[0,0].plot(distances, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
axs[0,0].plot(distances, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
axs[0,0].plot(distances, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
axs[0,0].plot(distances, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
axs[0,0].set_ylabel("Reflection relative error TE TIR")
axs[0,0].set_xlabel("Distance (nm)")
#axs[0,0].set_ylim([-0.001,.15])
#axs[0,0].set_yscale("log")
axs[0,0].legend()


axs[0,1].plot(distances, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
axs[0,1].plot(distances, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
axs[0,1].plot(distances, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
axs[0,1].plot(distances, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
axs[0,1].set_ylabel("Reflection relative error TM TIR")
axs[0,1].set_xlabel("Distance (nm)")
#axs[0,1].set_ylim([-0.001,.15])
#axs[0,1].set_yscale("log")
axs[0,1].legend()


axs[1,0].plot(distances, abs(ts_s_te-ts_a_te)/np.abs(ts_s_te), 'b-v', label="abeles", markersize=4)
axs[1,0].plot(distances, abs(ts_s_te-ts_dn_te)/np.abs(ts_s_te), 'r-o', label="D2N", markersize=4)
axs[1,0].plot(distances, abs(ts_s_te-ts_t_te)/np.abs(ts_s_te), 'g-^', label="T", markersize=4)
axs[1,0].set_ylabel("Transmission relative error TE TIR")
axs[1,0].set_xlabel("Distance (nm)")
axs[1,0].set_ylim([-0.001,.15])
#axs[1,0].set_yscale("log")
axs[1,0].legend()


axs[1,1].plot(distances, abs(ts_s_tm-ts_a_tm)/np.abs(ts_s_tm), 'b-v', label="abeles", markersize=4)
axs[1,1].plot(distances, abs(ts_s_tm-ts_dn_tm)/np.abs(ts_s_tm), 'r-o', label="D2N", markersize=4)
axs[1,1].plot(distances, abs(ts_s_tm-ts_t_tm)/np.abs(ts_s_tm), 'g-^', label="T", markersize=4)
axs[1,1].set_ylabel("Transmission relative error TM TIR")
axs[1,1].set_xlabel("Distance (nm)")
axs[1,1].set_ylim([-0.001,.15])
#axs[1,1].set_yscale("log")
axs[1,1].legend()
plt.tight_layout()
plt.show()


## Kretschmann

print("Prism :")
materials = [1, mat1**2, "Silver"]

incidence = np.arcsin(1/mat1)+0.3

stack = [1, 2, 0]

distances = np.arange(1, 101, 2)
if (unit == "um"):
    distances = distances*1e-3

rs_s_te = []
ts_s_te = []

rs_t_te = []
ts_t_te = []

rs_a_te = []
ts_a_te = []

rs_i_te = []

rs_dn_te = []
ts_dn_te = []

rs_s_tm = []
ts_s_tm = []

rs_t_tm = []
ts_t_tm = []

rs_a_tm = []
ts_a_tm = []

rs_i_tm = []

rs_dn_tm = []
ts_dn_tm = []

for dist in distances:

    ## Case 1: single layer, TE

    epaisseurs = np.concatenate(([0],[dist],[0]))
    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    multi_stack1 = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
    a = time()
    r_i, R_i = PM.coefficient_I(multi_stack1,wav,incidence,1)
    b = time()
    rs_i_tm.append(R_i)

rs_a_te = np.array(rs_a_te)
rs_a_tm = np.array(rs_a_tm)
ts_a_te = np.array(ts_a_te)
ts_a_tm = np.array(ts_a_tm)

rs_s_te = np.array(rs_s_te)
rs_s_tm = np.array(rs_s_tm)
ts_s_te = np.array(ts_s_te)
ts_s_tm = np.array(ts_s_tm)

rs_t_te = np.array(rs_t_te)
rs_t_tm = np.array(rs_t_tm)
ts_t_te = np.array(ts_t_te)
ts_t_tm = np.array(ts_t_tm)

rs_dn_te = np.array(rs_dn_te)
rs_dn_tm = np.array(rs_dn_tm)
ts_dn_te = np.array(ts_dn_te)
ts_dn_tm = np.array(ts_dn_tm)

rs_i_te = np.array(rs_i_te)
rs_i_tm = np.array(rs_i_tm)


fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10,10))
axs[0].plot(distances, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
axs[0].plot(distances, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
axs[0].plot(distances, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
axs[0].plot(distances, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
axs[0].set_ylabel("Reflection relative error TE prism")
axs[0].set_xlabel("Distance (nm)")
axs[0].set_ylim([-0.001,.25])
#axs[0].set_yscale("log")
axs[0].legend()


axs[1].plot(distances, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
axs[1].plot(distances, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
axs[1].plot(distances, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
axs[1].plot(distances, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
axs[1].set_ylabel("Reflection relative error TM prism")
axs[1].set_xlabel("Distance (nm)")
axs[1].set_ylim([-0.001,.25])
#axs[1].set_yscale("log")
axs[1].legend()

plt.tight_layout()
plt.show()

