import numpy as np
from context import PM
import matplotlib.pyplot as plt


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5**2, 2.+10.2j]
wav = 20
eps = 1e-10
pas = 0.001


print("Normal incidence:")
incidence = 0
nb_prob = 0
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)



if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)


if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)


if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)


if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
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
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
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
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with single interface and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with single interface and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with single interface and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with single interface and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,0, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,0, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,0, method="T", pas=pas)

if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TE")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TE")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TE")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TE")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)

r, t, R, T = PM.diff_coefficient(chose,wav,incidence,1, pas=pas)

r_ab, t_ab, R_ab, T_ab = PM.diff_coefficient(chose,wav,incidence,1, method="A", pas=pas)


r_t, t_t, R_t, T_t = PM.diff_coefficient(chose,wav,incidence,1, method="T", pas=pas)


if (np.sum(abs(r-r_ab))> eps):
    print("Problem with two layers and abeles coeff refl in TM")
    print(f"r = {r}, r_abeles={r_ab}")
    nb_prob+=1
if (np.sum(abs(r-r_t))> eps):
    print("Problem with two layers and TMatrix coeff refl in TM")
    print(f"r = {r}, r_TMat={r_t}")
    nb_prob+=1
if (np.sum(abs(t-t_ab))> eps):
    print("Problem with two layers and abeles coeff trans in TM")
    print(f"t = {t}, t_abeles={t_ab}")
    nb_prob+=1
if (np.sum(abs(t-t_t))> eps):
    print("Problem with two layers and TMatrix coeff trans in TM")
    print(f"t = {t}, t_TMat={t_t}")
    nb_prob+=1

if nb_prob:
    print()
    prob = True
    nb_prob = 0

if not(prob):
    print("OK!")
    print()
