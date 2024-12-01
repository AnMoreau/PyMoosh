
import numpy as np

material_list = [1.,[2.,1.2],"Si","Water"]
stack = [0,2,3,2]
thickness = [300, 200, 500, 200]

import PyMoosh as PM
print(PM.__version__)

thing = PM.Structure(material_list, stack, thickness)

si = thing.materials[2]
water = thing.materials[3]

refractive_index_glass = np.sqrt(si.get_permittivity(600))
epsilon = water.get_permittivity(600)
# You need to specify a wavelength at which to compute the permittivity, here 600 nm

print(np.sqrt(epsilon))

wavelength = 600
interface = PM.Structure([1.,2.25],[0, 1],[10*wavelength, 10*wavelength])

# Incidence angle
angle_inc = 0.
# Polarization
pol = 1.
[r,t,R,T] = PM.coefficient(interface,wavelength,angle_inc,pol)

print('Fresnel coefficient')
print(r)
print('Reflectance')
print(R)


# More complex use case
from PyMoosh.vectorized import angular

# For TE polarization
incidence, r, t, R, T = angular(interface, wavelength, 0., 0., 89., 200)
# For TM polarization, same incidence angles
incidence, r_p, t_p, R_p, T_p = angular(interface, wavelength, 1., 0., 89., 200)

# Visualization of the result
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

plt.figure(1)
plt.plot(incidence, R, label="TE polarisation")
plt.plot(incidence, R_p, label="TM polarisation")
plt.ylabel('Reflectance')
plt.ylim(0,1)
plt.legend()
plt.show()


from PyMoosh.vectorized import spectrum


interface = PM.Structure([1.,"Si"],[0, 1],[10*wavelength, 10*wavelength])
# For TE polarization
wavelength, r, t, R, T = spectrum(interface, 0, 0., 400., 800., 200)
# For TM polarization, same incidence angles
wavelength, r_p, t_p, R_p, T_p = spectrum(interface, 0, 1., 400., 800., 200)

# Visualization of the result
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

plt.figure(2)
plt.plot(wavelength, R, label="TE polarisation")
plt.plot(wavelength, R_p, label="TM polarisation")
plt.ylabel('Reflectance')
plt.ylim(0,1)
plt.legend()
plt.show()
