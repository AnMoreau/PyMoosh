import numpy as np
import PyMoosh as PM
import matplotlib.pyplot as plt
from time import time

## Bragg mirror with increasing number of layers
mat1 = 1.5
mat2 = 2


wav = 40.2

ep1 = wav/(4*mat1)
ep2 = wav/(4*mat2)

layers = np.arange(5, 131, 5)


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
    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,1)
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
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te), label="abeles")
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te), label="D2N")
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te), label="T")
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te), label="Impedance")
axs[0,0].set_title("Reflection error TE Normal incidence")
axs[0,0].set_xlabel("Nb Layers")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm), label="abeles")
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm), label="D2N")
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm), label="T")
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm), label="Impedance")
axs[0,1].set_title("Reflection error TM Normal incidence")
axs[0,1].set_xlabel("Nb Layers")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te), label="abeles")
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te), label="D2N")
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te), label="T")
axs[1,0].set_title("Transmission error TE Normal incidence")
axs[1,0].set_xlabel("Nb Layers")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm), label="abeles")
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm), label="D2N")
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm), label="T")
axs[1,1].set_title("Transmission error TM Normal incidence")
axs[1,1].set_xlabel("Nb Layers")
axs[1,1].legend()

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
    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,1)
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
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te), label="abeles")
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te), label="D2N")
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te), label="T")
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te), label="Impedance")
axs[0,0].set_title("Reflection error TE large incidence")
axs[0,0].set_xlabel("Nb Layers")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm), label="abeles")
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm), label="D2N")
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm), label="T")
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm), label="Impedance")
axs[0,1].set_title("Reflection error TM large incidence")
axs[0,1].set_xlabel("Nb Layers")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te), label="abeles")
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te), label="D2N")
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te), label="T")
axs[1,0].set_title("Transmission error TE large incidence")
axs[1,0].set_xlabel("Nb Layers")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm), label="abeles")
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm), label="D2N")
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm), label="T")
axs[1,1].set_title("Transmission error TM large incidence")
axs[1,1].set_xlabel("Nb Layers")
axs[1,1].legend()


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
    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,1)
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
axs[0,0].plot(layers, abs(rs_s_te-rs_a_te), label="abeles")
axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te), label="D2N")
axs[0,0].plot(layers, abs(rs_s_te-rs_t_te), label="T")
axs[0,0].plot(layers, abs(rs_s_te-rs_i_te), label="Impedance")
axs[0,0].set_title("Reflection error TE Intermediate incidence")
axs[0,0].set_xlabel("Nb Layers")
axs[0,0].legend()


axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm), label="abeles")
axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm), label="D2N")
axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm), label="T")
axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm), label="Impedance")
axs[0,1].set_title("Reflection error TM Intermediate incidence")
axs[0,1].set_xlabel("Nb Layers")
axs[0,1].legend()


axs[1,0].plot(layers, abs(ts_s_te-ts_a_te), label="abeles")
axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te), label="D2N")
axs[1,0].plot(layers, abs(ts_s_te-ts_t_te), label="T")
axs[1,0].set_title("Transmission error TE Intermediate incidence")
axs[1,0].set_xlabel("Nb Layers")
axs[1,0].legend()


axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm), label="abeles")
axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm), label="D2N")
axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm), label="T")
axs[1,1].set_title("Transmission error TM Intermediate incidence")
axs[1,1].set_xlabel("Nb Layers")
axs[1,1].legend()



plt.show()

## Frustrated total internal reflection


print("Frustrated total internal reflection:")
materials = [1, mat1**2, mat2**2]

incidence = np.arcsin(1/mat1)-0.01

stack = [1, 0, 1]

distances = wav * np.arange(0.01, 0.5, 0.01)

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
    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)
    b = time()
    rs_s_te.append(R)
    ts_s_te.append(T)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)
    b = time()
    rs_a_te.append(R_ab)
    ts_a_te.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)
    b = time()
    rs_t_te.append(R_t)
    ts_t_te.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)
    b = time()
    rs_dn_te.append(R_dn)
    ts_dn_te.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,0)
    b = time()
    rs_i_te.append(R_i)


    chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)
    b = time()
    rs_s_tm.append(R)
    ts_s_tm.append(T)



    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)
    b = time()
    rs_a_tm.append(R_ab)
    ts_a_tm.append(T_ab)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)
    b = time()
    rs_t_tm.append(R_t)
    ts_t_tm.append(T_t)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)
    b = time()
    rs_dn_tm.append(R_dn)
    ts_dn_tm.append(T_dn)


    chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
    a = time()
    r_i, R_i = PM.coefficient_I(chose1,wav,incidence,1)
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
axs[0,0].plot(distances, abs(rs_s_te-rs_a_te), label="abeles")
axs[0,0].plot(distances, abs(rs_s_te-rs_dn_te), label="D2N")
axs[0,0].plot(distances, abs(rs_s_te-rs_t_te), label="T")
axs[0,0].plot(distances, abs(rs_s_te-rs_i_te), label="Impedance")
axs[0,0].set_title("Reflection error TE frustrated TIR")
axs[0,0].set_xlabel("Distance")
axs[0,0].legend()


axs[0,1].plot(distances, abs(rs_s_tm-rs_a_tm), label="abeles")
axs[0,1].plot(distances, abs(rs_s_tm-rs_dn_tm), label="D2N")
axs[0,1].plot(distances, abs(rs_s_tm-rs_t_tm), label="T")
axs[0,1].plot(distances, abs(rs_s_tm-rs_i_tm), label="Impedance")
axs[0,1].set_title("Reflection error TM frustrated TIR")
axs[0,1].set_xlabel("Distance")
axs[0,1].legend()


axs[1,0].plot(distances, abs(ts_s_te-ts_a_te), label="abeles")
axs[1,0].plot(distances, abs(ts_s_te-ts_dn_te), label="D2N")
axs[1,0].plot(distances, abs(ts_s_te-ts_t_te), label="T")
axs[1,0].set_title("Transmission error TE frustrated TIR")
axs[1,0].set_xlabel("Distance")
axs[1,0].legend()


axs[1,1].plot(distances, abs(ts_s_tm-ts_a_tm), label="abeles")
axs[1,1].plot(distances, abs(ts_s_tm-ts_dn_tm), label="D2N")
axs[1,1].plot(distances, abs(ts_s_tm-ts_t_tm), label="T")
axs[1,1].set_title("Transmission error TM frustrated TIR")
axs[1,1].set_xlabel("Distance")
axs[1,1].legend()



plt.show()
