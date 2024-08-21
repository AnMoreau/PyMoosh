"""
This file contains easy-to-import functions for various
material models (Drude, Lorentz...)
"""

"""
From the old Material

        Models
            Drude case needs    : gamma0 (damping) and omega_p
            Drude-Lorentz needs : gamma0 (damping) and omega_p + f, gamma and omega of all resonance
            BB case needs       : gamma0 (damping) and omega_p + f, gamma, omega and sigma of all resonance

            - ExpData                / tuple(list(lambda), list(perm))   / 'ExpData'        / 'ExpData'
            - ExpData with mu        / as above with list(permeabilities)/ 'ExpData'        / 'ExpDataMu'
"""


"""


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

                

    def get_permittivity(self,wavelength):
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
        
        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)
        
"""