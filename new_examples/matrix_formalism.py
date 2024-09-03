import PyMoosh as PM

# PM.alt_methods DOES NOT WORK at this point (and shouldn't)
# but at least now you can import other methods
import PyMoosh.alt_methods as alt

material_list = [1.,1.5**2,"Water"]
stack = [0,2,1,0]
thickness=[0,500,500,0]

multilayer = PM.Structure(material_list,stack,thickness)
# Incidence angle
angle_inc=0.
# Polarization
pol=1.
# Wavelength
wavelength = 2.5
r, t, R, T = PM.coefficient(multilayer,wavelength,angle_inc,pol)

print(f"Reflection coefficient: {r}, Reflectance coefficient: {R}")
print(f"Transmission coefficient: {t}, Transmittance coefficient: {T}")


import numpy as np

#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5**2, 2.+1.j]
wav = 45.2


print("Intermediate incidence:")
incidence = 15*np.pi/180
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1.1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)

r_ab, t_ab, R_ab, T_ab = alt.coefficient_A(chose,wav,incidence,0)

r_t, t_t, R_t, T_t = alt.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = alt.coefficient_DN(chose,wav,incidence,0)

r_i, t_i, R_i, T_i = alt.coefficient_I(chose,wav,incidence,0)
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
structure = np.array([1.1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)

r_ab, t_ab, R_ab, T_ab = alt.coefficient_A(chose,wav,incidence,1)

r_t, t_t, R_t, T_t = alt.coefficient_T(chose,wav,incidence,1)

r_dn, t_dn, R_dn, T_dn = alt.coefficient_DN(chose,wav,incidence,1)

r_i, t_i, R_i, T_i = alt.coefficient_I(chose,wav,incidence,1)

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
structure = np.array([1.1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)

r_ab, t_ab, R_ab, T_ab = alt.coefficient_A(chose,wav,incidence,0)

r_t, t_t, R_t, T_t = alt.coefficient_T(chose,wav,incidence,0)

r_dn, t_dn, R_dn, T_dn = alt.coefficient_DN(chose,wav,incidence,0)

r_i, t_i, R_i, T_i = alt.coefficient_I(chose,wav,incidence,0)

print(f"two layers and abeles coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TE error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print(f"two layers and Impedance coeff trans in TE error = {np.format_float_scientific(abs(abs(t-t_i)/t), 2)}")

print()


## Case 4: two layers, TM
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1.1, 1.5])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,1)

r_ab, t_ab, R_ab, T_ab = alt.coefficient_A(chose,wav,incidence,1)

r_t, t_t, R_t, T_t = alt.coefficient_T(chose,wav,incidence,1)

r_dn, t_dn, R_dn, T_dn = alt.coefficient_DN(chose,wav,incidence,1)

r_i, t_i, R_i, T_i = alt.coefficient_I(chose,wav,incidence,1)


print(f"two layers and abeles coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_ab)/r), 2)}")


print(f"two layers and TMatrix coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_t)/r), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_dn)/r), 2)}")


print(f"two layers and Impedance coeff refl in TM error = {np.format_float_scientific(abs(abs(r-r_i)/r), 2)}")


print(f"two layers and abeles coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_ab)/t), 2)}")


print(f"two layers and TMatrix coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_t)/t), 2)}")


print(f"two layers and Dirichlet_to_Neumann coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_dn)/t), 2)}")

print(f"two layers and Impedance coeff trans in TM error = {np.format_float_scientific(abs(abs(t-t_i)/t), 2)}")

print()