import numpy as np
from context import PM
import matplotlib.pyplot as plt


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5**2+0.1j, 2.+0.2j]

unit="um"
wav = 0.200
eps = 1e-10


print("Normal incidence:")
incidence = 0*np.pi/180
nb_prob = 0
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3
lay = [0,1]

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]

print("1 layer TE")

epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

# print(a, R, T)
# print()
#
a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))
#
# print(a, R, T)
#
# #
a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)




if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TE")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0
#
# ## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3

print("1 layer TM")
# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]
#
#
# epaisseurs = np.concatenate(([0],structure,[0]))
#
# chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
#
a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))
print(a, R, T)
print()



a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)



if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
# structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

print("2 layers TE")

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)



if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs inTE.")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
#     print()
    prob = True
    nb_prob = 0
#
# ## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

print("2 layers TM")

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))
print(a, R, T)



a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)



if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
#     nb_prob = 0
#
# if not(prob):
#     print("OK!")
#     print()
#
#
print("Large incidence:")
incidence = 60*np.pi/180
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TE")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0
#
## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs inTE.")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
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
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TE")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)


if (np.sum(abs(r-r_a))> eps):
    print("Problem with single interface and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with single interface and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with single interface and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,0)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))

a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,0)

if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TE")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TE")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs inTE.")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([100, 150])
if (unit == "um"):
    structure = structure*1e-3

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)

a, r, t, R, T = PM.absorption_S(chose,wav,incidence,1)
if (R+T+np.sum(a)!=1):
    print("total energy (should be one)", R+T+np.sum(a))


a_a, r_a, t_a, R_a, T_a = PM.absorption_A(chose,wav,incidence,1)



if (np.sum(abs(r-r_a))> eps):
    print("Problem with two layers and Abeles coeff refl in TM")
    print(f"r = {r}, r_AMat={r_a}")
    nb_prob+=1

if (np.sum(abs(t-t_a))> eps):
    print("Problem with two layers and Abeles coeff trans in TM")
    print(f"t = {t}, t_AMat={t_a}")
    nb_prob+=1
if (np.sum(abs(a-a_a)>eps)):
    print("Problem with two layers and Abeles coeff abs in TM")
    print(f"a = {a}, a_AMat = {a_a}")
    nb_prob+=1
if nb_prob:
    print()
    prob = True
    nb_prob = 0

if not(prob):
    print("OK!")
    print()
