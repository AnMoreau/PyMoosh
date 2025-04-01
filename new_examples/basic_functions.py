
import numpy as np
import PyMoosh as PM

###############################################

### Setting up the studied structure
# Defining materials
material_list = [1.,"Si","Water"]

# Defining the vertical order of the materials
stack = [0,1,2,1]

# Defining the thickness of each layer
thickness = [300, 200, 500, 200]

# Defining the structure
struct = PM.Structure(material_list, stack, thickness)

### Simple calculations
wavelength = 700 # nm !

angle_inc = np.pi / 4 # rad

pol = 1 # 0 for TE, 1 for TM

# Calculation
r, t, R, T = PM.coefficient(struct,wavelength,angle_inc,pol)


### Spectrum
wav_beg = 400
wav_end = 800
nb_wav = 150

wavs, r, t, R, T = PM.spectrum(struct, wavelength, angle_inc, wav_beg, wav_end, nb_wav)

import matplotlib.pyplot as plt
plt.plot(wavs, R)
plt.xlabel("wavelength (nm)")
plt.ylabel("Reflectivity")
plt.show()



################################################
## More complex use cases

# Accessing materials directly
si = struct.materials[1]
water = struct.materials[2]

refractive_index_glass = np.sqrt(si.get_permittivity(600))
epsilon = water.get_permittivity(600)
# You need to specify a wavelength at which to compute the permittivity, here 600 nm

print(refractive_index_glass)
print(np.sqrt(epsilon))


## Studying a single interface

interface = PM.Structure([1.,2.25],[0, 1],[10*wavelength, 10*wavelength])

# Wavelength
wavelength = 600
# Incidence angle
angle_beg = 0.
angle_end = np.pi/6.
nb_angle = 200
# Polarization
pol = 1.

# For TE polarization
incidence, r, t, R, T = PM.angular(interface, wavelength, 0., angle_beg, angle_end, nb_angle, in_unit="rad")
# For TM polarization, same incidence angles
incidence, r_p, t_p, R_p, T_p = PM.angular(interface, wavelength, 1., angle_beg, angle_end, nb_angle, in_unit="rad")

# Visualization of the result
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(incidence, R, label="TE polarisation")
plt.plot(incidence, R_p, label="TM polarisation")
plt.ylabel('Reflectance')
# plt.ylim(0,1)
plt.legend()
plt.show()




# interface = PM.Structure([1.,"Si"],[0, 1],[10*wavelength, 10*wavelength])
# # For TE polarization
# wavelength, r, t, R, T = PM.spectrum(interface, 0, 0., 400., 800., 200)
# # For TM polarization, same incidence angles
# wavelength, r_p, t_p, R_p, T_p = PM.spectrum(interface, 0, 1., 400., 800., 200)

# # Visualization of the result
# import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 150

# plt.figure(2)
# plt.plot(wavelength, R, label="TE polarisation")
# plt.plot(wavelength, R_p, label="TM polarisation")
# plt.ylabel('Reflectance')
# plt.ylim(0,1)
# plt.legend()
# plt.show()
