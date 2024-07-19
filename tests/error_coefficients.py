import numpy as np
from context import PM
import matplotlib.pyplot as plt


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5**2, 2.+1.j]
wav = 45.2


print("WARNING: the impedance formalism only computes r and R for the moment")

print("Normal incidence")
incidence = 0
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)




print(f"single interface and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)

print(f"single interface and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"single interface and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")

print(f"single interface and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)

chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)

print(f"two layers and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)

print(f"two layers and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

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
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)

chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)
print(f"single interface and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"single interface and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)
print(f"single interface and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"single interface and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)

chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)
print(f"two layers and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)
print(f"two layers and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

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
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)
print(f"single interface and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"single interface and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 2: single layer, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)
print(f"single interface and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"single interface and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"single interface and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"single interface and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"single interface and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"single interface and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 3: two layers, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,0)

chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,0)
print(f"two layers and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()


## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t = PM.coefficient_T(chose1,wav,incidence,1)

chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(chose1,wav,incidence,1)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_i, t_i, R_i, T_i = PM.coefficient_I(chose1,wav,incidence,1)

print(f"two layers and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print()
