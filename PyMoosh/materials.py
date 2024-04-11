# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
#from enum import Enum

class Material:

    """
        Types:
            - simple_perm
            - magnetic
            - ExpData
            - CustomFunction
            - BrendelBormann
    """

    def __init__(self, mat, verbose=False):
        if mat.__class__.__name__ == 'function':
            self.type = "CustomFunction"
            self.permittivity_function = mat
            self.name = "CustomFunction: "+mat.__name__
            if verbose :
                print("Custom dispersive material. Epsilon=",mat.__name__,"(wavelength in nm)")
        elif not hasattr(mat, '__iter__'):
        # no func / not iterable --> single value, convert to complex by default
            self.type = "simple_perm"
            self.name = "SimplePermittivity:"+str(mat)
            self.permittivity = complex(mat)
            if verbose :
                print("Simple, non dispersive: epsilon=",self.permittivity)
        elif isinstance(mat,list) or isinstance(mat,tuple) or isinstance(mat,np.ndarray):
        # iterable: if list or similar --> magnetic
            self.type = "magnetic"
            self.permittivity = mat[0]
            self.permeability = mat[1]
            self.name = "MagneticPermittivity:"+str(mat[0])+"Permability:"+str(mat[1])
            if verbose :
                print("Magnetic, non dispersive: epsilon=", mat[0]," mu=",mat[1])
            if len(mat)>2:
                print(f'Warning: Magnetic material should have 2 values (epsilon / mu), but {len(mat)} were given.')
        elif isinstance(mat,str):
        # iterable: string --> database material
        # from file in shipped database
            import pkgutil
            f = pkgutil.get_data(__name__, "data/material_data.json")
            f_str = f.decode(encoding='utf8')
            database = json.loads(f_str)
            if mat in database:
                material_data = database[mat]
                model = material_data["model"]
                if model == "ExpData":
                    self.type = "ExpData"
                    self.name = "ExpData: "+ str(mat)

                    wl=np.array(material_data["wavelength_list"])
                    epsilon = np.array(material_data["permittivities"])
                    if "permittivities_imag" in material_data:
                        epsilon = epsilon + 1j*np.array(material_data["permittivities_imag"])

                    self.wavelength_list = np.array(wl, dtype=float)
                    self.permittivities  = np.array(epsilon, dtype=complex)

                elif model == "BrendelBormann":
                    self.type = "BrendelBormann"
                    self.name = "BrendelBormann model: " + str(mat)
                    self.f0 = material_data["f0"]
                    self.Gamma0 = material_data["Gamma0"]
                    self.omega_p = material_data["omega_p"]
                    self.f = np.array(material_data["f"])
                    self.gamma = np.array(material_data["Gamma"])
                    self.omega = np.array(material_data["omega"])
                    self.sigma = np.array(material_data["sigma"])

                elif model == "CustomFunction":
                    self.type = "CustomFunction"
                    self.name = "CustomFunction: " + str(mat)
                    permittivity = material_data["function"]
                    self.permittivity_function = authorized[permittivity]

                else:
                    print(model," not an existing model (yet).")
                    #sys.exit()

                if verbose :
                    print("Database material:",self.name)
            else:
                print(mat,"Unknown material (for the moment)")
                print("Known materials:\n", existing_materials())
                # sys.exit()

    def __str__(self):
        return self.name

    def get_permittivity(self,wavelength):
        if self.type == "simple_perm":
            return self.permittivity
        elif self.type == "magnetic":
            return self.permittivity
        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)
        elif self.type == "CustomFunction":
            return self.permittivity_function(wavelength)
        elif self.type == "BrendelBormann":
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

    def get_permeability(self,wavelength):
        if self.type == "magnetic":
            return self.permeability
        return 1.0

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
