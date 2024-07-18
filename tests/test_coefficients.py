import numpy as np
from context import PM
import matplotlib.pyplot as plt
import math


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [1, 1.2**2, 1.5**2 ]

unit = "nm"
wav = 600
eps = 1e-12
nb = 10


print("WARNING: the impedance formalism only computes r and R for the moment")

print("Normal incidence:")
incidence = 0
nb_prob = 0
prob = False

single_structure = np.array([100, 0])
complex_structure = np.array([wav/(4*1.2), wav/(4*1.5)]*nb + [140, 0])
if (unit == "um"):
    single_structure = single_structure*1e-3
    complex_structure = complex_structure*1e-3

single_stack = [0,1, 2]

complex_stack = [0] + [1,2]*nb + [1, 0]
## Case 1: single layer, TE

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)

if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)


if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TM")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)


if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)


if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TM")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

if not(prob):
    print("OK!")
    print()


print("Large incidence:")
incidence = 60*np.pi/180
prob = False

## Case 1: single layer, TE

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)

if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)

if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TM")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)

if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)

if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

if not(prob):
    print("OK!")
    print()


print("Intermediate incidence:")
incidence = 15*np.pi/180
prob = False

## Case 1: single layer, TE

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)

if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM

stack = single_stack


epaisseurs = np.concatenate(([0],single_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)


r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)

if (abs(r-r_ab)> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with single interface and Dirichlet_to_Neumann coeff refl in TM")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with single interface and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with single interface and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with single interface and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,0)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,0)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,0)

if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TE")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TE")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TE")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TE")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM

stack = complex_stack


epaisseurs = np.concatenate(([0],complex_structure))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose,wav,incidence,1)


r_t, t_t, R_t, T_t = PM.coefficient_T(chose,wav,incidence,1)

r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose,wav,incidence,1)


r_i, t_i, R_i, T_i = PM.coefficient_I(chose,wav,incidence,1)


if (abs(r-r_ab)> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (abs(r-r_t)> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (abs(r-r_dn)> eps or math.isnan(np.real(r_dn))):
    print("Problem with two layers and Dirichlet_to_Neumann coeff refl in TM")
    print(f"r = {r}, r_D2N={r_dn}")
    nb_prob+=1
if (abs(r-r_i)> eps):
    print("Problem with two layers and Impedance coeff refl in TM")
    print(f"r = {r}, r_I={r_i}")
    nb_prob+=1
if (abs(t-t_ab)> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (abs(t-t_i)> eps):
    print("Problem with two layers and impedance coeff trans in TM")
    print(f"t = {t}, t_impedance={t_i}")
    nb_prob+=1
if (abs(t-t_t)> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1
if (abs(t-t_dn)> eps):
    print("Problem with two layers and Dirichlet_to_Neumann coeff trans in TM")
    print(f"t = {t}, t_D2N={t_dn}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

if not(prob):
    print("OK!")
    print()
