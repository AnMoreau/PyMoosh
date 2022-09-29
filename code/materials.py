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

class Interpolation(Material):
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

    def get_permeability(self, wavelength):
        return 1.0

class Simple_non_dispersive(Material):

    def __init__(self,permittivity):
        self.permittivity = permittivity

    def get_permittivity(self,wavelength):
        return self.permittivity

    def get_permeability(self,wavelength):
        return 1.0


class Magnetic_non_dispersive(Material):

    def __init__(self,permittivity,permeability):

        self.permittivity = permittivity
        self.permeabilty  = permeability

    def get_permittivity(self,wavelength):
        return self.permittivity

    def get_permeability(self,wavelength):
        return self.permeability


class BBmodel(Material):
    """
    Material described using a Brendel Bormann model for a metal.
    """

    def __init__(self, material_data: dict) -> None:
        self.name = material_data["name"]
        self.material_data = material_data
        self._custom = False

    def get_permittivity(self, **kwargs):
        wavelength = kwargs["wavelength"]
        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength
        # Extract material properties
        f0 = self.material_data["f0"]
        gamma0 = self.material_data["Gamma0"]
        # Cast to numpy array for computation
        omega_p = np.array(self.material_data["omega_p"])
        f = np.array(self.material_data["Gamma0"])
        gamma = np.array(self.material_data["Gamma"])
        omega = np.array(self.material_data["omega"])
        sigma = np.array(self.material_data["sigma"])
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
    def get_permeability(self, **kwargs):
        return 1.0




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



class MaterialEnum(Enum):
    CUSTOM = {"name": "Custom", "class": CustomMaterial}
    AIR = {"name": "Air", "class": Air}
    BK_7 = {"name": "BK7", "class": BK7}
    WATER = {"name": "Water", "class": Water}
    GLASS = {"name": "Glass", "class": Glass}
    SIA = {"name": "SiA", "class": SiA}
    SILVER = {"name": "Silver", "class": Silver}
    GOLD = {"name": "Gold", "class": Gold}
    PLATINUM = {"name": "Platinum", "class": Platinum}
    NICKEL = {"name": "Nickel", "class": Nickel}
    COPPER = {"name": "Copper", "class": Copper}


def MaterialFactory(material: MaterialEnum):
    """
    Factory class for the creation of materials
    """
    # Get the name of the material from Enum
    material_name = material.value["name"]
    if material_name != "Custom":
        # Load the json file with material information
        json_filepath = "./../data/material_data.json"
        f = open(str(json_filepath))
        material_database = json.load(f)
        # Get the desired material given its name
        material_data = material_database[material_name]
        # Create the class with proper
        material_class = material.value["class"](material_data)
    # If the material is custom
    else:
        material_class = material.value["class"]
    # Return the appropriate class with the material data to be used
    return material_class
