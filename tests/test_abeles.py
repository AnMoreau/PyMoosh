import numpy as np
import PyMoosh as PM
import matplotlib.pyplot as plt


incidence = 10/180*np.pi
polarisation = 0
#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5, 1.]
w_mean = 1

nb_couches = 0

wav = 1

rs = []
r_abs = []
for i in range(1):
    #structure = np.random.random(nb_couches*2+1)*w_mean
    structure = np.array([1, 1])

    # stack = [0]+[1,2]*nb_couches+[1,0]
    stack = [0,2, 1,0]


    epaisseurs = np.concatenate(([0],structure,[0]))

    print(structure, stack)

    chose = PM.Structure(materials,stack,epaisseurs, verbose=True)
    r, t, R, T = PM.coefficient(chose,wav,incidence,polarisation)

    r_ab, t_ab, R_ab, T_ab = PM.abeles_coefficients(chose,wav,incidence,polarisation)


    chose = PM.Structure(materials,stack,epaisseurs, verbose=True)
    r_t, t_t, R_t, T_t = PM.TMatrix_coefficients(chose,wav,incidence,polarisation)


    print("POP")
    print(r, r_ab, r_t)
    print(t, t_ab, t_t)
    print("DIFFS")
    print(r- r_ab, r-r_t)
    print(t-t_ab, t-t_t)
    print(R, R_ab, R_t)
    print(T, T_ab, T_t)
#    rs.append(r)
#    r_abs.append(r_ab)

#plt.plot(np.real(rs), label="RE r")
#plt.plot(np.imag(rs), label="RE r")
#plt.plot(np.real(r_abs), label="RE r_abeles")
#plt.plot(np.imag(r_abs), label="IM r_abeles")
#plt.legend()

#plt.figure()
#plt.plot(np.real(rs)**2, label="RE r")
#plt.plot(np.imag(rs)**2, label="RE r")
#plt.plot(np.real(r_abs)**2, label="RE r_abeles")
#plt.plot(np.imag(r_abs)**2, label="IM r_abeles")
#plt.show()
