# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
#from enum import Enum

class Material:

    def __init__(self,permittivity, name=""):
        self.permittivity = permittivity
        self.name = name

    def __str__(self):
        if (len(self.name) > 1):
            return f"{self.name}, perm: {self.permittivity}"
        return str(self.permittivity)

    def get_permittivity(self,wavelength):
        return self.permittivity

    def get_permeability(self,wavelength):
        return 1.0

class CustomFunction(Material):

    def __init__(self,permittivity_function, name=""):
        self.permittivity_function = permittivity_function
        self.name = name

    def __str__(self):
        if (len(self.name) > 1):
            return f"{self.name}, model: custom function, see material_data.json"
        return "Custom function for permittivity"

    def get_permittivity(self,wavelength):
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

    def __init__(self, wavelength_list,permittivities, name=""):

        self.wavelength_list = np.array(wavelength_list, dtype=float)
        self.permittivities  = np.array(permittivities, dtype=complex)
        self.name = name

    def __str__(self):
        if (len(self.name) > 1):
            return f"{self.name}, model: Exp Data"
        return "Experimental Data for database material"

    def get_permittivity(self, wavelength):
        return np.interp(wavelength, self.wavelength_list, self.permittivities)

class MagneticND(Material):

    """
    Magnetic, non-dispersive material, characterized by a permittivity
    and a permeabilty that do not depend on the wavelength.
    """

    def __init__(self,permittivity,permeability):
        self.permittivity = permittivity
        self.permeability  = permeability

    def get_permeability(self,wavelength):
        return self.permeability


class BrendelBormann(Material):
    """
    Material described using a Brendel Bormann model for a metal.
    """

    def __init__(self, f0,gamma0,omega_p,f,gamma,omega,sigma, name="") -> None:
        self.f0 = f0
        self.Gamma0 = gamma0
        self.omega_p = omega_p
        self.f = np.array(f)
        self.gamma = np.array(gamma)
        self.omega = np.array(omega)
        self.sigma = np.array(sigma)
        self.name = name


    def __str__(self):
        if (len(self.name) > 1):
            return f"{self.name}, model: Brendel Bormann"
        return "Brendel Bormann model for Database material"

    def get_permittivity(self, wavelength):
        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength
        a = np.sqrt(w * (w + 1j * self.gamma))
        x = (a - self.omega) / (np.sqrt(2) * self.sigma)
        y = (a + self.omega) / (np.sqrt(2) * self.sigma)
        # Polarizability due to bound electrons
        chi_b = np.sum(1j * np.sqrt(np.pi) * self.f * self.omega_p ** 2 /
                       (2 * np.sqrt(2) * a * self.sigma) * (wofz(x) + wofz(y)))
        # Equivalent polarizability linked to free electrons (Drude model)
        chi_f = -self.omega_p ** 2 * self.f0 / (w * (w + 1j * self.Gamma0))
        epsilon = 1 + chi_f + chi_b
        return epsilon

def existing_materials():
    import pkgutil
    f = pkgutil.get_data(__name__, "data/material_data.json")
    f_str = f.decode(encoding='utf8')
    database = json.loads(f_str)
    for entree in database:
        if "info" in database[entree]:
            print(entree,"::",database[entree]["info"])
        else :
            print(entree)

# Sometimes materials can be defined not by a well known model
# like Cauchy or Sellmeier or Lorentz, but have specific formula.
# That may be convenient.

def permittivity_glass(wl):
    #epsilon=2.978645+0.008777808/(wl**2*1e-6-0.010609)+84.06224/(wl**2*1e-6-96)
    epsilon = (1.5130 - 3.169e-9*wl**2 + 3.962e3/wl**2)**2
    return epsilon

# Declare authorized functions in the database. Add the functions listed above.

authorized = {"permittivity_glass":permittivity_glass}
