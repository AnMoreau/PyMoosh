import numpy as np
from context import PM
import matplotlib.pyplot as plt
import itertools

mat = 1.5
#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [1., mat**2, 2]


print("Normal incidence:")
incidence = 0*np.pi/180
nb_prob = 0

structure = np.array([2000, 500])
epaisseurs = np.concatenate(([0],structure,[0]))
stack = [0,1, 2, 0]

wavs = np.linspace(400, 600, 100)
neff_min, neff_max = 1., 2.


chose = PM.Structure(materials, stack, epaisseurs)

indices, follow_modes = PM.follow_guided_modes(chose, wavs, 0, neff_min, neff_max, format="n")
