# encoding utf-8
import numpy as np
from scipy.special import wofz
import json
from refractiveindex import RefractiveIndexMaterial
#from PyMoosh.anisotropic_functions import get_refraction_indices

class Material:

    """
        Types of material (default):
              type                   / format:                 / specialType:

            - material               / Material object         / 'Default'
            - CustomFunction         / function (wav)          / 'Default'
            - simple_perm            / complex                 / 'Default'
            - magnetic               / list(complex, float)    / 'Default'
            - Database               / string                  / 'Default'
                  Database types can take any special types from below

        The Default types can be used directly when initialising the structure

        Types of material (special): / format:                           / specialType keyword / detectable self.type:

            - CustomModel            / list(function, parameters)        / 'Model'          / 'Model'
            - BB Model               / list(parameters)                  / 'BrendelBormann' / 'BrendelBormann'
            - Drude Model            / list(parameters)                  / 'Drude'          / 'Drude' or 'DrudeLorentz
            - ExpData                / tuple(list(lambda), list(perm))   / 'ExpData'        / 'ExpData'
            - ExpData with mu        / as above with list(permeabilities)/ 'ExpData'        / 'ExpDataMu'
            - RefractiveIndexInfo    / list(shelf, book, page)           / 'RII'            / 'RefractiveIndexInfo'
            - Anisotropic            / list(no, ne) or list(n1, n2, n3)  / 'ANI'            / 'Anisotropic'
            - Anisotropic from RII   / list(shelf, book, page)           / 'ANI_RII'        / 'Anisotropic'

        Models
            Drude case needs    : gamma0 (damping) and omega_p
            Drude-Lorentz needs : gamma0 (damping) and omega_p + f, gamma and omega of all resonance
            BB case needs       : gamma0 (damping) and omega_p + f, gamma, omega and sigma of all resonance


        Non local materials
            - custom function based / function   and params         / 'NonLocal'       / 'NonLocalModel'

            All non local materials need: beta0, tau, omegap
            + all the parameters needed in their respective models
            custom function must return: beta2, chi_b, chi_f, omega_p

        Special types must be processed through the Material constructor before being passed on to Structure as a Material object
        Models included are "BrendelBormann" "Drude" "DrudeLorentz" [TODO]
    """

    def __init__(self, mat, specialType="Default", verbose=False):

        if issubclass(mat.__class__,Material):
            # Has already been processed by this constructor previously
            if verbose:
                print("Preprocessed material:", mat.__name__)

        if specialType == "Default":
            # The default behaviors listed in the docstring
            self.specialType = specialType
            if mat.__class__.__name__ == 'function':
            # Is a custom function that only takes the wavelength as a parameter
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
                    
            elif isinstance(mat, list) and isinstance(mat[0], float) and isinstance(mat[1], float): # magnetic == [float, float]
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
            # iterable: string --> database material from file in shipped database
                import pkgutil
                f = pkgutil.get_data(__name__, "data/material_data.json")
                f_str = f.decode(encoding='utf8')
                database = json.loads(f_str)
                if mat in database:
                    material_data = database[mat]
                    model = material_data["model"]
                    
                    if model == "ExpData":
                        # Experimnental data to be interpolated
                        self.type = "ExpData"
                        self.name = "ExpData: "+ str(mat)

                        wl=np.array(material_data["wavelength_list"])
                        epsilon = np.array(material_data["permittivities"])
                        if "permittivities_imag" in material_data:
                            epsilon = epsilon + 1j*np.array(material_data["permittivities_imag"])

                        self.wavelength_list = np.array(wl, dtype=float)
                        self.permittivities  = np.array(epsilon, dtype=complex)
                    
                    if model == "BrendelBormann":
                        # Brendel & Bormann model with all necessary parameters
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
                        # Database custom function taking only the wavelength as argument
                        self.type = "CustomDatabaseFunction"
                        self.name = "CustomDatabaseFunction: " + str(mat)
                        permittivity = material_data["function"]
                        self.permittivity_function = authorized[permittivity]

                    else:
                        print(model," not an existing model (yet).")
                        #sys.exit()

                    if verbose :
                        print("Database material:",self.name)
                else:
                    print(mat,"Unknown material in the database (for the moment)")
                    #print("Known materials:\n", existing_materials())
                    #sys.exit()

            else:
                print(f'Warning: Given data is not in the right format for a \'Default\' specialType. You should check the data format or specify a specialType. You can refer to the following table:')
                print(self.__doc__)

        elif specialType == "RII":
            # Refractive index material, anisotropic
            if len(mat) != 3:
                print(f'Warning: Material RefractiveIndex Database is expected to be a list of 3 values, but {len(mat)} were given.')
            self.type = "RefractiveIndexInfo"
            self.specialType = specialType
            self.name = "MaterialRefractiveIndexDatabase: " + str(mat)
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(shelf, book, page) # not necessary ?
            material = RefractiveIndexMaterial(shelf, book, page) # create object
            self.material = material
            if verbose :
                # print("Hello there ;)")
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')

        elif specialType == "BrendelBormann" :
            # Brendel-Bormann model with n resonances
            self.type = "BrendelBormann"
            self.specialType = specialType
            self.name = "BrendelBormann model : " + str(mat)
            self.f0 = mat[0]
            self.Gamma0 = mat[1]
            self.omega_p = mat[2]
            if (len(mat[3])==len(mat[4])==len(mat[5])==len(mat[6])):
                self.f = np.array(mat[3])
                self.gamma = np.array(mat[4])
                self.omega = np.array(mat[5])
                self.sigma = np.array(mat[6])
            if verbose:
                print("Brendel Bormann model for material with parameters ", str(mat))

        elif specialType == "Drude" or specialType == "DrudeLorentz" :
            # Simple Drude-Lorentz  model
            self.type = "Drude"
            self.specialType = specialType
            self.name = "Drude model : " + str(mat)
            self.Gamma0 = mat[0]
            self.omega_p = mat[1]
            if len(mat) == 5 and (len(mat[2])==len(mat[3])==len(mat[4])):
                self.type = "DrudeLorentz"
                self.f = np.array(mat[2])
                self.gamma = np.array(mat[3])
                self.omega = np.array(mat[4])
            if verbose:
                print("Drude(Lorentz) model for material with parameters ", str(mat))

        elif specialType == "Lorentz" :
            # Simple Lorentz model
            self.type = "Lorentz"
            self.specialType = specialType
            self.name = "Lorentz model : " + str(mat)
            self.eps = mat[0]
            if len(mat) == 4 and (len(mat[1])==len(mat[2])==len(mat[3])):
                self.type = "DLorentz"
                self.f = np.array(mat[1])
                self.gamma = np.array(mat[2])
                self.omega = np.array(mat[3])
            if verbose:
                print("Drude(Lorentz) model for material with parameters ", str(mat))

        elif specialType == "ExpData":
            # Experimental Data given as a list of wavelengths and permittivities
            self.type = "ExpData"
            self.name = "ExpData: "+ str(mat)
            self.specialType = specialType

            self.wavelength_list = np.array(mat[0], dtype=float)
            self.permittivities  = np.array(mat[1], dtype=complex)
            if len(mat) == 3:
                self.type = "ExpDataMu"
                self.permeabilities  = np.array(mat[2], dtype=complex)

        elif specialType == "ANI" :
            # User defined Anisotropic material
            if len(mat) < 2 or len(mat) > 3:
                print(f'Warning: Anisotropic material is expected to be a list of 2 or 3 index values, but {len(mat)} were given.')
            self.type = "Anisotropic"
            self.specialType = specialType
            if (len(mat) == 2):
                # Uniaxial, only two values given, no and ne
                self.material_x = mat[0]
                self.material_y = mat[0]
                self.material_z = mat[1]
            elif (len(mat) == 3):
                # Biaxial, three values given,
                self.material_x = mat[0]
                self.material_y = mat[1]
                self.material_z = mat[2]

            self.name = "Anisotropic material" + str(mat)
            if verbose :
                print("Anisotropic material of indices ", str(mat))
                
        elif specialType == "ANI_RII" :
            # Anisotropic material from the refractive index database
            if len(mat) != 3:
                print(f'Warning: Anisotropic material from Refractiveindex.info is expected to be a list of 3 values, but {len(mat)} were given.')
            self.type = "Anisotropic"
            self.specialType = specialType
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(shelf, book, page) # not necessary ?
            material_list = wrapper_anisotropy(shelf, book, page) # A list of three materials
            self.material_list = material_list
            self.material_x = material_list[0]
            self.material_y = material_list[1]
            self.material_z = material_list[2]
            self.name = "Anisotropic material from Refractiveindex.info: " + str(mat)
            if verbose :
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(f'Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given.')
        
        elif specialType == "Model":
            # A custom function that takes more parameters than simply the wavelength
            self.type = 'Model'
            self.specialType = specialType
            self.permittivity_function = mat[0]
            self.params = [mat[i+1] for i in range(len(mat)-1)]
            self.name = "Customfunction: " + str(mat[0])

        elif specialType == "NonLocal" : 
            self.specialType = "NonLocal"
            # Non local material defined as a function for the parameters
            # The function must follow the following workings:
            # returns beta2, chi_b, chi_f and omega_p (in this order)
            if  mat[0].__class__.__name__ != "function" :
                print("Please provide a function for the model, or used default Drude/BB models")
            else:
                self.type = "NonLocalModel"
                self.name = "NonLocalModel : " + str(mat[0])
                self.NL_function = mat[0]
                self.params = [mat[i+1] for i in range(len(mat)-1)]
                if verbose :
                    print("Custom non local dispersive material defined by function ", str(self.NL_function))

        # elif specialType == "NL" or specialType == "NLDrude" :
        #     # Non local material defined by a Drude model for the chi_f
        #     self.SpecialType = "NL"
        #     self.type = "NonLocalDrude"
        #     self.beta0 = self.mat[0]
        #     self.tau = self.mat[1]
        #     self.omega_p = self.mat[2]
        #     self.chi_b = self.mat[3]
        #     self.gamma = self.mat[4]
        #     self.name = "NonLocalDrude :" + str(mat)
            
        #     self.Gamma0 = mat[0]
        #     self.omega_p = mat[1]
        #     if len(mat) == 5 and (len(mat[2])==len(mat[3])==len(mat[4])):
        #         self.type = "DrudeLorentz"
        #         self.f = np.array(mat[2])
        #         self.gamma = np.array(mat[3])
        #         self.omega = np.array(mat[4])

        #     if verbose :
        #         print(f"NonLocalMaterial : [chi_b = {self.chi_b}, chi_f = {self.chi_f}, w_p = {self.w_p}, beta = {self.beta}] SpecialType = {self.SpecialType}")


        #elif specialType == "NLBB" : # WIP
            # Non local material defined by a Brendel Borman model for the chi_f
            # self.SpecialType = "NL"
            # self.beta0 = self.mat[0]
            # self.tau = self.mat[1]
            # self.omega_p = self.mat[2]
            # self.chi_b = self.mat[3]
            # self.gamma = self.mat[4]
            # self.f0 = mat[0]
            # self.Gamma0 = mat[1]
            # self.omega_p = mat[2]
            # if (len(mat[3])==len(mat[4])==len(mat[5])==len(mat[6])):
            #     self.f = np.array(mat[3])
            #     self.gamma = np.array(mat[4])
            #     self.omega = np.array(mat[5])
            #     self.sigma = np.array(mat[6])
                          
            
        elif specialType == "Unspecified":
            self.specialType = specialType
            print(specialType, "Unknown type of material (for the moment)")
            # sys.exit()

        else:
            print(f'Warning: Unknown type : {specialType}')


    def __str__(self):
        return self.name


    def get_permittivity(self,wavelength):
        if self.type == "simple_perm":
            return self.permittivity
        
        elif self.type == "magnetic":
            return self.permittivity
        
        elif self.type == "CustomFunction":
            return self.permittivity_function(wavelength)
        
        elif self.type == "Model":
            return self.permittivity_function(wavelength, *self.params)
        
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
        
        elif self.type == "Drude":
            # TODO: decide if omega_p / gamma are in Hz or rad s-1 !!
            w = 2*np.pi*299792458*1e9 / wavelength
            chi_f = - self.omega_p ** 2 / (w * (w + 1j * self.Gamma0))
            return 1 + chi_f
        
        elif self.type == "Lorentz":
            w = 2*np.pi*299792458*1e9 / wavelength
            chi = np.sum(self.f/(self.omega**2 - w**2 - 1.0j*self.gamma*w))
            return self.eps + chi
        
        elif self.type == "DrudeLorentz":
            w = 2*np.pi*299792458*1e9 / wavelength
            chi_f = - self.omega_p ** 2 / (w * (w + 1j * self.Gamma0))
            chi_b = np.sum(self.f/(self.omega**2 - w**2 - 1.0j*self.gamma*w))
            return 1 + chi_f + chi_b
        
        elif self.type == "RefractiveIndexInfo":
            try:
                k = self.material.get_extinction_coefficient(wavelength)
                return self.material.get_epsilon(wavelength)
            except:
                n = self.material.get_refractive_index(wavelength)
                return n**2
        
        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)
        
        elif self.type == "Anisotropic":
            print(f'Warning: Functions for anisotropic materials generaly requires more information than isotropic ones. You probably want to use \'get_permittivity_ani()\' function.')
        
        elif self.specialType == "NonLocal":
            _, chi_b, chi_f, _ = self.get_values_nl(wavelength)
            return 1 + chi_b + chi_f
           


    def get_permeability(self,wavelength, verbose=False):
        if self.type == "magnetic":
            return self.permeability
        elif self.type == "RefractiveIndexInfo":
            if verbose:
                print('Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 .')
            return 1.0
        elif self.type == "Anisotropic":
            if verbose:
                print('Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 .')
            return [1.0, 1.0, 1.0]
        elif self.type == "ExpDataMu":
            return np.interp(wavelength, self.wavelength_list, self.permeabilities)
        return 1.0
    

    # def old_get_values_nl(self, wavelength = 500) :
    #     # Retrieving the non local material parameters

    #     if self.type == "NonLocalModel" :
    #         self.beta = self.mat(wavelength)[3]
    #         self.w_p = self.mat(wavelength)[2]
    #         self.chi_b = self.mat(wavelength)[0]
    #         self.chi_f = self.mat(wavelength)[1]
        
    #     elif self.type == "NonLocalMaterial" :
    #         self.beta = self.mat[3]
    #         self.w_p = self.mat[2]
    #         self.chi_b = self.mat[0]
    #         self.chi_f = self.mat[1]

    #     return self.chi_b, self.chi_f, self.w_p, self.beta
    
    def get_values_nl(self, wavelength = 500) :
        # Retrieving the non local material parameters

        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength

        if self.type == "NonLocalModel" :
            res = self.NL_function(wavelength, *self.params)
            beta2 = res[0]
            chi_b = res[1]
            chi_f = res[2]
            omega_p = res[3]
        
        # elif self.type == "NonLocalDrude" :
        #     beta2 = self.beta0**2 + 1.0j * w * self.tau
        #     chi_b = self.omega_p
        #     # chi_f = #INSERT DRUDE MODEL HERE
        #     omega_p = self.omega_p
        
        # elif self.type == "NonLocalBrendelBormann" :
        #     beta2 = self.beta0**2 + 1.0j * w * self.tau
        #     chi_b = self.omega_p
        #     # chi_f = #INSERT BB MODEL HERE
        #     omega_p = self.omega_p

        else:
            print("You're not supposed to be here: get_values_nl with no known NL function defined")

        return beta2, chi_b, chi_f, omega_p
    

# Anisotropic method
    """ def get_permittivity_ani(self, wavelength, elevation_beam, precession, nutation, spin):
        # We have three permittivities to extract
        refraction_indices_medium = []
        for material in self.material_list:# A complex refractive index is denoted m=n+ik. However, in the Refractive index database  
            try:                           # n and k are only given separately by "get_refractive_index" and "get_extinction_coefficient" respectivly.
                k = material.get_extinction_coefficient(wavelength)
                refraction_indices_medium.append(material.material.get_epsilon(wavelength))# Here we use the fact that "get_epsilon(wl)" return an error if k is not given in the Ref Ind dataB.  
            except:
                n = material.get_refractive_index(wavelength)
                refraction_indices_medium.append(n**2)
        return np.sqrt(get_refraction_indices(elevation_beam, refraction_indices_medium, precession, nutation, spin))"""
    
#AV# Here i just need to get the permittivity of the material but this function does much more than getting the permittivity so i make mine: 
    def get_permittivity_ani(self, wavelength):
        epsilon_medium = []
        for material in self.material_list:
            try:
                k = material.get_extinction_coefficient(wavelength)
                epsilon_medium.append(material.get_epsilon(wavelength)) # Here we use the fact that "get_epsilon(wl)" return an error if k is not given in the Ref Ind dataB to go in the except where we deal with the real index case. 
                print('k =',k )
                print('epsilon_medium =' ,material.get_epsilon(wavelength)) 
            except:                                                              # If k exist we use get_epsilon(wl) 
                n = material.get_refractive_index(wavelength)
                epsilon_medium.append(n**2)
                print('n =',n)
        return epsilon_medium
'''Reminder : this function can handle the case of complex n thanks to 
def get_epsilon(self, wavelength_nm, exp_type='exp_minus_i_omega_t'):
        n = self.get_refractive_index(wavelength_nm)
        k = self.get_extinction_coefficient(wavelength_nm)
        if exp_type=='exp_minus_i_omega_t':
            return (n + 1j*k)**2
        else:
            return (n - 1j*k)**2 '''


def wrapper_anisotropy(shelf, book, page):
    if page.endswith("-o") or page.endswith("-e"):
        if page.endswith("-e"):
            page_e, page_o = page, page.replace("-e", "-o")
        elif page.endswith("-o"):
            page_e, page_o = page.replace("-o", "-e"), page

        # create ordinary and extraordinary object.
        material_o = RefractiveIndexMaterial(shelf, book, page_o)
        material_e = RefractiveIndexMaterial(shelf, book, page_e)
        return [material_o, material_o, material_e]
    
    elif page.endswith("-alpha") or page.endswith("-beta") or page.endswith("-gamma"):
        if page.endswith("-alpha"):
            page_a, page_b, page_c = page, page.replace("-alpha", "-beta"), page.replace("-alpha", "-gamma")
        elif page.endswith("-beta"):
            page_a, page_b, page_c = page.replace("-beta", "-alpha"), page, page.replace("-beta", "-gamma")
        elif page.endswith("-gamma"):
            page_a, page_b, page_c = page.replace("-gamma", "-alpha"), page.replace("-gamma", "-beta"), page
        
        # create ordinary and extraordinary object.
        material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
        material_beta = RefractiveIndexMaterial(shelf, book, page_b)
        material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
        return [material_alpha, material_beta, material_gamma]
    
    else:
        # there may better way to do it.
        try:
            page_e, page_o = page + "-e", page + "-o"
            material_o = RefractiveIndexMaterial(shelf, book, page_o)
            material_e = RefractiveIndexMaterial(shelf, book, page_e)
            return [material_o, material_o, material_e]
        except:
            try:
                page_a, page_b, page_c = page + "-alpha", page + "-beta", page + "-gamma"
                print(page_a)
                material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
                print(material_alpha)
                material_beta = RefractiveIndexMaterial(shelf, book, page_b)
                print(material_beta)
                material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
                print(material_gamma)
                return [material_alpha, material_beta, material_gamma]
            except:
                print(f'Warning: Given material is not known to be anisotropic in the Refractiveindex.info database. You should try to remove "ANI" keyword in material definition or to spellcheck the given path.')


def permittivity_glass(wl):
    #epsilon=2.978645+0.008777808/(wl**2*1e-6-0.010609)+84.06224/(wl**2*1e-6-96)
    epsilon = (1.5130 - 3.169e-9*wl**2 + 3.962e3/wl**2)**2
    return epsilon

# Declare authorized functions in the database. Add the functions listed above.

authorized = {"permittivity_glass":permittivity_glass}
