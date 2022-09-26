# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
from enum import Enum


class CustomMaterial:
    """
    Class for user defined material.
    """

    def __init__(self, epsilon: complex = 1.0, mu: complex = 1.0,
                 name: str = "custom") -> None:
        self.name = name
        self.epsilon = epsilon
        self.mu = mu
        self._custom = True

    def get_permittivity(self, **kwargs):
        return self.epsilon

    def get_permeability(self, **kwargs):
        return self.mu


class NonConductingMaterial:
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

class ConductingMaterial:
    """
    Generic class to handle conducting material
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
    def get_permeability(self):
        return 1.0

class BK7(NonConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Water(NonConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Air(NonConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)

    def get_permittivity(self, **kwargs) -> float:
        """
        Overides NonConductingMaterial mother class method
        """
        return 1.0


class Glass(NonConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)

    def get_permittivity(self, **kwargs) -> float:
        """
        Overides NonConductingMaterial mother class method
        """
        wavelength = kwargs["wavelength"]
        epsilon = 2.978645 + 0.008777808 / \
                  (wavelength ** 2 * 1e-6 - 0.010609) + 84.06224 / (
                          wavelength ** 2 * 1e-6 - 96)
        return epsilon


class SiA(NonConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Silver(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Aluminium(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Gold(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Nickel(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Platinum(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


class Copper(ConductingMaterial):
    def __init__(self, material_data: dict) -> None:
        super().__init__(material_data)


# Enum of available materials


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


if __name__ == '__main__':
    # Get a material from the material Factory
    material = MaterialFactory(MaterialEnum.GOLD)
    custom = MaterialFactory(MaterialEnum.CUSTOM)(epsilon=1.5, mu=1)
    # Get the material name
    print(material.name)
    print(material.epsilon)
    # Get the permittivity at a specific lambda
    print(material.get_permittivity_at_wavelength(150.))
