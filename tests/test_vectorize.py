import numpy as np
from context import PM
from PyMoosh import vectorized as vec
from PyMoosh.models import Drude
import matplotlib.pyplot as plt
import math

mat1 = PM.Material([Drude, 1e10, 1e5], specialType="Model")
mat2 = PM.Material(["main", "SiO2", "Malitson"], specialType="RII")
materials = [1.**2, "Gold", mat2, mat1]
# materials = [1, 1.2**2, 1.5**2 ]

incidence = 0
nb_prob = 0
prob = False


single_stack = [0,3, 2, 1]

## Case 1: single layer, TE

stack = single_stack


epaisseurs = np.array([0, 100, 150, 0])

interface = PM.Structure(materials,stack,epaisseurs, verbose=True, si_units=True)


wavelength, sr, st, sR, sT = vec.spectrum_S(interface, 0, 0., 400., 800., 20)
# # For TM polarization, same incidence angles
# wavelength, sr_p, st_p, sR_p, sT_p = vec.spectrum_S(interface, 0, 1., 400., 800., 200)

# wavelength, ar, at, aR, aT = vec.spectrum_A(interface, 0, 0., 400., 800., 200)
# # For TM polarization, same incidence angles
# wavelength, ar_p, at_p, aR_p, aT_p = vec.spectrum_A(interface, 0, 1., 400., 800., 200)

# # Visualization of the result
# import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 150

# plt.figure(2)
# plt.plot(wavelength, np.real(sr), label="Smat TE polarisation")
# plt.plot(wavelength, np.real(sr_p), label="Smat TM polarisation")
# plt.plot(wavelength, np.real(ar), "+", label="Amat TE polarisation")
# plt.plot(wavelength, np.real(ar_p), "+", label="Amat TM polarisation")
# plt.ylabel('r')
# plt.ylim(-1,1)
# plt.legend()
# plt.show()


wavelength = 600
interface = PM.Structure([1.,2.25],[0, 1],[10*wavelength, 10*wavelength])

# For TE polarization
angles, sr, st, sR, sT = vec.angular_S(interface, wavelength, 0., 0., 89., 20)
# For TM polarization, same incidence angles
angles, sr_p, st_p, sR_p, sT_p = vec.angular_S(interface, wavelength, 1., 0., 89., 20)

angles, ar, at, aR, aT = vec.angular_A(interface, wavelength, 0, 0., 89., 20)
# For TM polarization, same incidence angles
angles, ar_p, at_p, aR_p, aT_p = vec.angular_A(interface, wavelength, 1., 0., 89., 20)

# Visualization of the result
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

plt.figure(2)
plt.plot(angles, np.abs(sR), label="Smat TE polarisation")
plt.plot(angles, np.abs(sR_p), label="Smat TM polarisation")
plt.plot(angles, np.abs(aR), "+", label="Amat TE polarisation")
plt.plot(angles, np.abs(aR_p), "+", label="Amat TM polarisation")
plt.ylabel('r')
# plt.ylim(-1,1)
plt.legend()
plt.show()
