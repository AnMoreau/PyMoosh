import numpy as np
import PyMoosh as pm
import matplotlib.pyplot as plt
import itertools


def Landau(freq, mudc, fres, m):
    w0 = 2*np.pi*fres
    g = m*w0
    w = 2*np.pi*freq
    mur = 1 + (mudc-1)*(w0**2 + 1j*w*g)/(w0**2 - w**2 + 2j*w*g)
    return mur



def mu(freq):
    omega = 2 * np.pi * freq
    omega_0 = 2 * np.pi * 1e6
    gamma = 0.7*omega_0
    Mu_dc = 30
    return 1 + (Mu_dc - 1) * (omega_0**2 + 1j*omega*gamma)/(omega_0**2 - omega**2 + 2j*omega*gamma)



def mu_nm(wav):
    freq = 6.62606957e-25 * 299792458 / wav
    omega = 2 * np.pi * freq
    omega_0 = 2 * np.pi * 1e6
    gamma = 0.7*omega_0
    Mu_dc = 30
    muc = 1 + (Mu_dc - 1) * (omega_0**2 + 1j*omega*gamma)/(omega_0**2 - omega**2 + 2j*omega*gamma)
    return np.real(muc) -1.0j*np.imag(muc) 


def eps(wav):
    return 1.5


mat_mu = [[eps],[mu_nm]]  # 2 lists, one each of the format [function, params]
mat = pm.Material(mat_mu, specialType="ModelMu")
materials = [1., mat, 2]

structure = np.array([2000, 500])
epaisseurs = np.concatenate(([0],structure,[0]))
stack = [0,1, 2, 0]


freq = np.logspace(3, 8, 201)
wavs = 6.62606957e-25 * 299792458 / freq

# print(mu(freq))
plt.plot(freq, np.real(mu_nm(wavs)))
plt.plot(freq, -np.imag(mu(freq)))
plt.xscale("log")
plt.show()


chose = pm.Structure(materials, stack, epaisseurs)
Rs = []
for i, f in enumerate(freq):
    _, _, R, _ = pm.coefficient(chose, wavs[i], 10*np.pi/180.,1)
    Rs.append(R)
Rs = np.array(Rs)
plt.plot(freq, Rs)
plt.xscale("log")
plt.show()