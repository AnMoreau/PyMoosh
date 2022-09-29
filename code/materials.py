# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
from enum import Enum

class Material:

    def __init__(self,permittivity):
        self.permittivity = permittivity

    def get_permittivity(self,wavelength):
        return self.permittivity

    def get_permeability(self,wavelength):
        return 1.0

class CustomFunction(Material):

    def __init__(self,permittivity_function)
        self.permittivity_function = permittivity_function

    def get_permittivity(self,wavlength)
        return self.permittivity_function(wavelength)

class ExpData(Material):
    """
    Class of materials defined by their permittivity measured for
    well defined values of the wavelength in vacuum. We make asin
    interpolation to get the most accurate value of the permittivity.
    Two lists are thus expected:
    - wavelength_list
    - permittivities (potentially complex)
    """

    def __init__(self, wavelength_list,permittivities):

        self.wavelength_list = np.array(wavelength_list, dtype = float)
        self.permittivities  = np.array(permittivities, dtype = complex)

    def get_permittivity(self, wavelength):
        return np.interp(wavelength, wavelength_list, permittivity_list)

class MagneticND(Material):

    """
    Magnetic, non-dispersive material, characterized by a permittivity
    and a permeabilty that do not depend on the wavelength.
    """

    def __init__(self,permittivity,permeability):

        self.permittivity = permittivity
        self.permeabilty  = permeability

    def get_permeability(self,wavelength):
        return self.permeability


class BrendelBormann(Material):
    """
    Material described using a Brendel Bormann model for a metal.
    """

    def __init__(self, f0,gamma0,omega_p,f,gamma,omega,sigma) -> None:
        self.f0 = f0
        self.Gamma0 = gamma0
        self.omega_p = omega_p
        self.f = np.array(f)
        self.gamma = np.array(gamma)
        self.omega = np.array(omega)
        self.sigma = np.array(sigma)

    def get_permittivity(self, wavelength):
        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength
        # Extract material properties
        # Compute variables
        a = np.sqrt(w * (w + 1j * gamma))
        x = (a - omega) / (np.sqrt(2) * sigma)
        y = (a + omega) / (np.sqrt(2) * sigma)
        # Polarizability due to bound electrons
        chi_b = np.sum(1j * np.sqrt(np.pi) * f * omega_p ** 2 /
                       (2 * np.sqrt(2) * a * sigma) * (wofz(x) + wofz(y)))
        # Equivalent polarizability linked to free electrons (Drude model)
        chi_f = -omega_p ** 2 * f0 / (w * (w + 1j * gamma0))
        epsilon = 1 + chi_f + chi_b
        return epsilon

class NonConductingMaterial(Material):
    """
    Generic class to handle non-conducting material.
    """

    def __init__(self, material_data: dict) -> None:
        self.name = material_data["name"]
        self.material_data = material_data
        self._custom = False

    def get_permittivity(self, **kwargs) -> float:
        """
        Get the permittivity using interpolation
        """
        wavelength = kwargs["wavelength"]
        wavelength_list = self.material_data["wavelengths"]
        permittivity_list = self.material_data["permittivities_real"]
        # Get the complex permittivity if needed
        if "permittivity_imag" in self.material_data.keys():
            permittivity_cmplx = self.material_data["permittivity_imag"]
            # Loop over both list to reconstruct complex permittivity
            permittivity_list = [esp_real + 1j * eps_imag for esp_real,
                                                              eps_imag in
                                 zip(permittivity_list, permittivity_cmplx)]

        return np.interp(wavelength, wavelength_list, permittivity_list)

    def get_permeability(self,**kwargs) -> float:
        return 1.0

def Existing_materials():
    f=open("../data/material_data.json")
    database = json.load(f)
    print(database.values())
