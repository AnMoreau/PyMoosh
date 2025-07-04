"""
This file contains easy-to-import functions for various
material models (Drude, Lorentz, Brendel&Bormann...)
"""

import numpy as np
from scipy.special import erfc

"""
From the old Material

        Models
            Drude case needs    : gamma0 (damping) and omega_p
            Drude-Lorentz needs : gamma0 (damping) and omega_p + f, gamma and omega of all resonance
            BB case needs       : gamma0 (damping) and omega_p + f, gamma, omega and sigma of all resonance
            ExpData                / tuple(list(lambda), list(perm))   / 'ExpData'        / 'ExpData'
"""


def BrendelBormann(wav, f0, omega_p, Gamma0, f, omega, gamma, sigma):
    """
    Brendel & Bormann model, using Voigt functions to model lorentzian
    resonances potentially widened with a gaussian distribution.
    f0, Gamma0 and omega_p are the chi_f parameters (eps_inf, plasma frequency)
    f, gamma, omega, sigma are the chi_b parameters (Lorentz resonances)
    f, gamma, omega, sigma must be lists (np arrays) of the same lengths
    They are given in eV (wav in nm)
    """
    # Brendel-Bormann model with n resonances
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    chi_b = 0
    for i in range(len(f)):
        a = np.sqrt(w * (w + 1j * gamma[i]))
        x = (a - omega[i]) / (np.sqrt(2) * sigma[i])
        y = (a + omega[i]) / (np.sqrt(2) * sigma[i])
        # Polarizability due to bound electrons
        erx = np.exp(-(x**2)) * erfc(-1.0j * x)
        ery = np.exp(-(y**2)) * erfc(-1.0j * y)
        oscill_strength = (
            1j * np.sqrt(np.pi) * f[i] * omega_p**2 / (2 * np.sqrt(2) * a * sigma[i])
        )
        chi_b += oscill_strength * (erx + ery)
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f = -(omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon


def Drude(wav, omega_p, Gamma0):
    """
    Drude model, with only the plasma frequency omega_p
    and damping Gamma0 and omega_p
    They are given in eV (wav in nm)
    """
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    chi_f = -(omega_p**2) / (w * (w + 1j * Gamma0))
    return 1 + chi_f


def Lorentz(wav, f, omega, gamma, eps):
    """
    Lorentz model, with lorentzian resonances (elastically bound electrons)
    eps is eps_inf, the background permittivity
    f, gamma, omega, sigma are the chi_b parameters (Lorentz resonances)
    f, gamma, omega, sigma must be lists (np arrays) of the same lengths
    They are given in eV (wav in nm)
    """
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    chi = 0
    for i in range(len(f)):
        chi += f[i] / (omega[i] ** 2 - w**2 - 1.0j * gamma[i] * w)
    return eps + chi


def DrudeLorentz(wav, omega_p, Gamma0, f, omega, gamma):
    """
    Drude Lorentz model, using both lorentzian resonances and a plasma frequency
    f0, Gamma0 and omega_p are the chi_f parameters (eps_inf, plasma frequency)
    f, gamma, omega, sigma are the chi_b parameters (Lorentz resonances)
    f, gamma, omega, sigma must be lists (np arrays) of the same lengths
    They are given in eV (wav in nm)
    """
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    chi_f = -(omega_p**2) / (w * (w + 1j * Gamma0))
    chi_b = 0
    for i in range(len(f)):
        chi_b += f[i] / (omega[i] ** 2 - w**2 - 1.0j * gamma[i] * w)
    return 1 + chi_f + chi_b


def ExpData(wav, wav_list, permittivities):
    """
    Interpolation from experimental data
    Wavelenght must be in nm
    """
    return np.interp(wav, wav_list, permittivities)
