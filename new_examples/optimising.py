import PyMoosh as PM
import numpy as np

# The wavelength we are interested in
wav = 600
# The angle of incidence and polarization the structure is intended for
angle = 25 * np.pi / 180
polar = 0 # 0 for TE, 1 for TM

def objective_function(layers, wavelength=wav, angle=angle, polar=polar):
    """
    We want to maximise the reflectance of the structure for the chosen wavelength
    """
    nb_lay = len(layers)//2
    mat = [1, 1.5, 2]
    stack = [0] + [1, 2] * nb_lay + [0]
    thickness = [0] + [t for t in layers] + [0]
    structure = PM.Structure(mat, stack, thickness, verbose=False)
    r, t, R, T = PM.coefficient(structure, wavelength, angle, polar)
    return 1-R


budget = 1000 #

nb_layers = 10 # We want a 10-layer Dielectric mirror
min_lay = 10 # No layer thinner than 10 nm
max_lay = 300 # No layer thicker than 800 nm (half the largestthe wavelength)

X_min = np.array([min_lay] * nb_layers)
X_max = np.array([max_lay] * nb_layers)

best, convergence = PM.differential_evolution(objective_function, budget, X_min, X_max)

import matplotlib.pyplot as plt

plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Cost function")
plt.show()


wav_beg = 400
wav_end = 800
nb_wav = 100
reflectivity = np.zeros(nb_wav)
wav_list = np.linspace(wav_beg, wav_end, nb_wav)
mat = [1, 1.5, 2]
stack = [0] + [1, 2] * (nb_layers//2) + [0]
thickness = [0] + [t for t in best] + [0]
structure = PM.Structure(mat, stack, thickness, verbose=False)
for i, wav in enumerate(wav_list):
    r, t, R, T = PM.coefficient(structure, wav, angle, polar)
    reflectivity[i] = R

plt.plot(wav_list, reflectivity)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectivity")
budget = 15000

nb_layers = 30 # We want a 30-layer Dielectric mirror
min_lay = 10 # No layer thinner than 10 nm
max_lay = 300 # No layer thicker than 800 nm (half the wavelength)

X_min = np.array([min_lay] * nb_layers)
X_max = np.array([max_lay] * nb_layers)

best, convergence = PM.differential_evolution(objective_function, budget, X_min, X_max)

plt.plot(convergence)
plt.xlabel("Iteration")
plt.ylabel("Cost function")
plt.show()



reflectivity = np.zeros(nb_wav)
wav_list = np.linspace(wav_beg, wav_end, nb_wav)
mat = [1, 1.5, 2]
stack = [0] + [1, 2] * (nb_layers//2) + [0]
thickness = [0] + [t for t in best] + [0]
structure = PM.Structure(mat, stack, thickness, verbose=False)
for i, wav in enumerate(wav_list):
    r, t, R, T = PM.coefficient(structure, wav, angle, polar)
    reflectivity[i] = R

plt.plot(wav_list, reflectivity)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectivity")