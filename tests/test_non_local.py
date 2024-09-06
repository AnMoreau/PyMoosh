import numpy as np
from context import PM
import matplotlib.pyplot as plt
import csv
from PyMoosh.non_local import *
#Xbest =  [11.6644913, 8.9e+14, 3.53201120e+12,  1.92864068e+30,  9.718*10**14]
#        [chi_b,      w_p,     gamma,           beta_0,          tau         ]

wavelength_list = np.linspace(5000, 8000, 3001)

with open('nlplot.data', newline='') as csvfile:
    plot = list(csv.reader(csvfile))[0]
plot = [float(p.strip()) for p in plot]
plot = np.array(plot)

"""
def wrapper_exemple(P, wl) :
    fonction de conversion entre les elements donnees sous la forme [chi_b, w_p, gamma, beta_p, beta_s, gamma, base, scale] a [chi_b, chi_f, w_p, beta]

    #[chi_b, w_p, gamma, beta_p, beta_s, base, scale]
    w = 2 * np.pi * 299792458 / (wl * 10**(-9))
    chi_f = -P[1]**2 / (w * (w + 1j * P[2]))

    #[chi_b, chi_f, w_p, beta]
    return P[0], chi_f, P[1], np.sqrt(P[3] - 1j * P[4] * w)
"""

def nl_function(wavelength, chi_b):
    w_p = 8.9e+14
    gamma = 3.53201120e+12
    beta_0 = np.sqrt(1.92864068e+30)
    tau = 9.718*10**14

    w = 2*np.pi*299792458*1e9 / wavelength
    beta2 = beta_0**2 - 1.0j * w * tau
    chi_f = - w_p**2/(w * (w + 1j * gamma))

    return beta2, chi_b, chi_f, w_p

mat_non_local = NLMaterial([nl_function, 11.6644913])
materials = [14.2129, 15.6816, mat_non_local]

stack = [1, 2, 0]
thickness = [0, 105, 0]
theta = np.pi * 37 / 180
pol = 1.0

chose = NLStructure(materials,stack,thickness, verbose=False)

wl, r, t, R, T = PM.spectrum(chose, theta, pol, 5000, 8000, 300)

import matplotlib.pyplot as plt

plt.plot(wavelength_list, plot, label="ref")
plt.plot(wl, R*0.2, label="R plot with parameters de tic et tac")
plt.legend()
plt.show()

# Test de la permittivité du modèle.
# eps = []
# for w in wl:
#     beta2, chi_b, chi_f, w_p = nl_function(w, 11.6644913)
#     eps.append(1+chi_b+chi_f)
# eps = np.array(eps)
# plt.plot(wl, eps)
# plt.ylabel("Eps(NL)")
# plt.show()


#%% Optimization


wavelength_list = np.linspace(5000, 8000, 3001)
# plot =

nb_lam = 50

def nl_function(wavelength, chi_b, w_p, gamma, beta_0, tau):

    w = 2*np.pi*299792458*1e9 / wavelength
    beta2 = beta_0**2 - 1.0j * w * tau
    chi_f = - w_p**2/(w * (w + 1j * gamma))

    return beta2, chi_b, chi_f, w_p


def cost_function(X):
    chi_b, w_p, gamma, beta_0, tau, base, scale = X
    mat_non_local = PM.Material([nl_function, chi_b, w_p, gamma, beta_0, tau], specialType="NonLocal")
    materials = [14.2129, 15.6816, mat_non_local]

    stack = [1, 2, 0]
    thickness = [0, 105, 0]
    theta = np.pi * 37 / 180
    pol = 1.0

    chose = PM.Structure(materials,stack,thickness, verbose=False)

    wl, r, t, R, T = PM.spectrum(chose, theta, pol, 5000, 8000, nb_lam)
    R = R*scale + base
    new_wl = np.linspace(5000, 8000, nb_lam)
    obj  = np.interp(new_wl, wavelength_list, plot)
    cost = np.mean(np.abs(R - obj))+10*np.mean(np.abs(np.diff(R)-np.diff(obj)))
    return cost/nb_lam

budget = 2000 #
nb_runs = 1


X_min = np.array([11.4, 1e14, 1e12, 1e14, 5e14, 0, 0.01])
X_max = np.array([11.8, 1e15, 1e13, 1.5e15, 1.2e15, 1, 2])

bests = []
convergences = []

for i in range(nb_runs):
    print("RUN ", i+1 ,"/", nb_runs)
    best, convergence = PM.QODE(cost_function, budget, X_min, X_max, progression=10)
    bests.append(best)
    convergences.append(convergence)

chi_b, w_p, gamma, beta_0, tau, base, scale = best
mat_non_local = PM.Material([nl_function, chi_b, w_p, gamma, beta_0, tau], specialType="NonLocal")
materials = [14.2129, 15.6816, mat_non_local]

stack = [1, 2, 0]
thickness = [0, 105, 0]
theta = np.pi * 37 / 180
pol = 1.0

chose = PM.Structure(materials,stack,thickness, verbose=False)
wl, r, t, R, T = PM.spectrum(chose, theta, pol, 5000, 8000, 500)
R = R*scale + base
print(f"best parameters found : chi_b={chi_b:e}, w_p={w_p:e}, gamma={gamma:e}, beta_0={beta_0:e}, tau={tau:e}")
plt.plot(np.linspace(5000,8000,500), R, label="optimized")
plt.plot(wavelength_list, plot, label="ref")
plt.legend()
plt.show()
