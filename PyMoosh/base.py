"""
This file contains all the basic structure necessary for PyMoosh to run
"""

import numpy as np
from numpy import linalg as la_np
import copy
import re
import matplotlib.pyplot as plt


def conv_to_nm(length, unit):
    """ Converts a length from "unit"to nm, because everything has been coded
        in nm...
    """
    if (unit == "m"):
        return np.array(length) * 1e9
    elif (unit == "um"):
        return np.array(length) * 1e3
    elif (unit == "mm"):
        return np.array(length) * 1e6
    elif (unit == "pm"):
        return np.array(length) * 1e-3
    elif (unit == "nm"):
    # Just in case we get here but didn't need to
        return np.array(length)
    else:
        print("Please provide lengths in m, mm, um, pm or nm")



def rotate_permittivity(eps, angle_rad, axis='z'): #AV#Aded
    """
    This function calculates the rotated permittivity tensor of eps around the given axis about the required angle

    Args :
    eps : permittivity tensor: a 3x3 Numpy array
    angle_rad : rotation angle (in radians) around the rotation axis ``axis``
    axis : rotation axis as a one-dimensional Numpy array of length 3 or a string  'x', 'y' or 'z'
    return : rotated permittivity tensor: a 3x3 Numpy array
    """

    # Retrieve the rotation axis
    # vec_u = unit vector
    x_u = np.array([1, 0, 0])
    y_u = np.array([0, 1, 0])
    z_u = np.array([0, 0, 1])
    if isinstance(axis, str):
        if axis == 'x':
            axis = x_u
        elif axis == 'y':
            axis = y_u
        elif axis == 'z':
            axis = z_u
        else:
            raise Exception('Invalid rotation axis.')
    if la_np.norm(axis) == 0: #if axis has zero norm then necessarily: axis=[0,0,0]
        raise Exception('Invalid axis. Axis can not be (0, 0, 0).')
    if np.array(axis).shape != x_u.shape :
        raise Exception('axis as to be a one-dimensional Numpy array of lenght 3')

    # Rotation matrix
    # theta_rad = theta_rad % (2 * np.pi)
    axis = axis / la_np.norm(axis) # axis normalisation
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    costheta = np.cos(angle_rad)
    sintheta = np.sin(angle_rad)
    R = np.array([[costheta + (ux ** 2) * (1 - costheta), ux * uy * (1 - costheta) - uz * sintheta,
                        ux * uz * (1 - costheta) + uy * sintheta],
                        [uy * ux * (1 - costheta) + uz * sintheta, costheta + (uy ** 2) * (1 - costheta),
                        uy * uz * (1 - costheta) - ux * sintheta],
                        [uz * ux * (1 - costheta) - uy * sintheta, uz * uy * (1 - costheta) + ux * sintheta,
                        costheta + (uz ** 2) * (1 - costheta)]])
    # Rotate permittivity tensor
    eps_rot = la_np.multi_dot((R, eps, R.transpose()))
    return eps_rot




class Structure:
    """Each instance of Structure describes a multilayer completely.
    This includes the materials the multilayer is made of and the
    thickness of each layer. If there is at least one anisotropic material
    in the stack, Structure also contain the list of rotation angle and rotation axis to orient the permittivity eigenbasis in each layer.

    Args:
        materials (list) : a list of material definition
        layer_type (list) : how the different materials are stacked
        thickness (list) : thickness of each layer in nm
        ani_rot_angle (list) : rotation angle for each layer used to define the orientation of the medium's eigenbasis
        ani_rot_axis (list) : rotation axis for each layer used to define the orientation of the medium's eigenbasis
        units (str) : the length unit used for thickness

    Materials can be defined in the list :materials:
    -by giving their permittivity as a real number for non dispersive materials
    -by giving a list of two floats. The first is the permittivity and the
    second one the permeability. It is considered non dispersive.
    -by giving its name for dispersive materials that are in the database.
    -by giving a custom permittivity function, taking as an argument the
    wavelength in nanometer.

    .. warning: the working wavelength always is given in nanometers.

    Example: [1.,'Si','Au'] means we will use a material with a permittivity
    of 1. (air), silicon and gold.

    Each material can then be refered to by its position in the list :materials:
    The list layer_type is used to describe how the different materials are
    placed in the structure, using the place of each material in the list
    :materials:

    Example: [0,1,2,0] describes a superstrate made of air (permittivity 1.),
    on top of Si, on top of gold and a substrate made of air.

    The thickness of each layer is given in the :thickness: list, in nanometers
    typically. The thickness of the superstrate is assumed to be zero by most
    of the routines (like :coefficient:, :absorption:) so that the first
    interface is considered as the phase reference. The reflection coefficient
    of the structure will thus never be impacted by the thickness of the first
    layer.For other routines (like :field:), this thickness define the part
    of the superstrate (or substrate) that must be represented on the figure.

    Example: [0,200,300,500] actually refers to an infinite air superstrate but
    non of it will be represented, a 200 nm thick silicon layer, a 300 nm thick
    gold layer and an infinite substrate, of which a thickness of 500 nm will be
    represented is asked.

    In the case of materials with anisotropic relative permittivity, it is necessary
    to define for each layer the orientation of the material's eigenbase (where the
    permittivity tensor is diagonal). This orientation is defined in relation to our
    reference base (in which the z axis is normal to the interface and the x axis
    lies in the plane of incidence). The rotation matrix used to express the
    permittivities in our reference basis from the medium's proper permittivities
    is defined by :

        a single rotation angle as an int or a float

        a rotation axis as a string ('x','y' or 'z') or a 3-row array (this vector will be automatically normalized)

    The list :ani_rot_angle: contain the rotation angles associated to each layer of the stack in :layer_type:.
    Similarly, the list :ani_rot_axis: contain the rotation axis associated to each layer of the stack in :layer_type:.
    To maintain the logic used to report the layer characteristics, we want the sizes of these two lists to match :layer_type:.
    So even if a layer is isotropic, it must have an angle and an axis assigned to it (which, in any case, will not affect its permittivity).

    """

    def __init__(self, materials, layer_type, thickness, ani_rot_angle=None, ani_rot_axis=None, verbose=True, unit="nm", si_units=False):

        if (unit != "nm"):
            thickness = conv_to_nm(thickness, unit)
            if not(si_units):
                print("I can see you are using another unit than nanometers, ",
                        "please make sure you keep using that unit everywhere.",
                        " To suppress this message, add the keyword argument si_units=True when you call Structure")

        self.unit = unit

        self.Anisotropic = False
        self.NonLocal = False

        materials_final=list()
        if verbose :
            print("List of materials:")
        for mat in materials:
            if issubclass(mat.__class__,Material):
                materials_final.append(mat)
                if mat.type == "Anisotropic":
                    self.Anisotropic = True
                elif mat.specialType == "NonLocal":
                    self.NonLocal = True
                if verbose :
                    print("Object:",mat.__class__.__name__)
            else :
                new_mat = Material(mat, verbose=verbose)
                materials_final.append(new_mat)
        self.materials = materials_final
        self.layer_type = layer_type
        self.thickness = thickness


        if self.Anisotropic:
            if ani_rot_angle == None:   # Setting all the angles to 0 if nothing has been specified by the user.
                self.ani_rot_angle = [0]*np.size(layer_type)

            else :
                self.ani_rot_angle = ani_rot_angle  # ani_rot_angle is a list of angle in radian (float or int). One for each layer, setting isotropic layers to the default angle = 0

            for ang in self.ani_rot_angle: # Checking format.
                if not(isinstance(ang, float) or isinstance(ang, int)):
                    raise Exception("angle have to be a float or a int")


            if ani_rot_axis == None: # Setting all the axis to 'z' if nothing has been specified by the user.
                self.ani_rot_axis = ['z']*np.size(layer_type)
            else :
                self.ani_rot_axis = ani_rot_axis# ani_rot-axis is a list of axis reprensented as a row array of length 3
                                                #or as the string ``'x'``, ``'y'`` or ``'z'``. One for each layer in the stack, setting isotropic layers to the default axis 'z'

            for ax in self.ani_rot_axis: # Checking format.
                if not(isinstance(ax, str)) and np.shape(ax) != np.shape([0,0,0]) :
                    raise Exception("axis have to be a string ``'x'``, ``'y'`` or ``'z'` or a row array of length 3")

            # Checking if the first and last layers are isotrop (Superstrate and Substrate are halfspaces respectively
            #representing the medium of the incoming and outgoing light of the multi-layer stack)
            if materials_final[layer_type[0]].type == "Anisotropic" or materials_final[layer_type[-1]].type == "Anisotropic":
                raise Exception("Superstrate's and Substrate's material have to be isotropic !")

        if self.NonLocal:
            if materials_final[layer_type[0]].specialType == "NonLocal" or materials_final[layer_type[-1]].specialType == "NonLocal":
                raise Exception("Superstrate's and Substrate's material have to be local !")



    def __str__(self):
        materials = [str(self.materials[i]) for i in range(len(self.materials))]
        s = f"materials: {materials}\nlayers: {self.layer_type}\nthicknesses: {self.thickness}"
        return s

    def polarizability(self, wavelength):
        """ Computes the actual permittivity and permeability of each material considered in
        the structure. This method is called before each calculation.

        Args:
            wavelength (float): the working wavelength (in nanometers)
        """


        # Create empty mu and epsilon arrays
        mu = np.ones_like(self.materials, dtype=complex)
        epsilon = np.ones_like(self.materials, dtype=complex)
        # Loop over all materials
        for k in range(len(self.materials)):
            # Populate epsilon and mu arrays from the material.
            material = self.materials[k]
            epsilon[k] = material.get_permittivity(wavelength)
            mu[k] = material.get_permeability(wavelength)

        return epsilon, mu


    def polarizability_opti_wavelength(self, wavelength): #numpy friendly
        """ ** Only used for coefficient_S_opti_wavelength **
        Computes the actual permittivity and permeability of each material considered in
        the structure. This method is called before each calculation.

        Args:
            wavelength (numpy array): the working wavelength (in nanometers)
        """


        # Create empty mu and epsilon arrays
        mu = np.ones((wavelength.size, len(self.materials)), dtype=np.clongdouble)
        epsilon = np.ones((wavelength.size, len(self.materials)), dtype=np.clongdouble)
        # Loop over all materials
        for k in range(len(self.materials)):
            # Populate epsilon and mu arrays from the material.
            material = self.materials[k]
            material_get_permittivity = material.get_permittivity(wavelength)
            material_get_permeability = material.get_permeability(wavelength)
            try:
                material_get_permittivity.shape = (len(wavelength),)
                material_get_permeability.shape = (len(wavelength),)
            except:
                pass

            epsilon[:,k] = material_get_permittivity
            mu[:,k] = material_get_permeability

        return epsilon, mu


    def permittivity_tensor_list(self, wavelength, layer=None):#AV_Added#
        """ Return the permittivity tensor of each material considered in the structure as a row array of 3*3 array. Both isotropic and anisotropic materials are supported.

         Args:
         wavelength (float): the working wavelength (in nanometers)
        """
        if layer is not None:
            Id_3 = np.eye(3 , 3)
            mat_lay_i = self.materials[self.layer_type[layer]]
            if mat_lay_i.type != "Anisotropic":
                return mat_lay_i.get_permittivity(wavelength)*Id_3
            else :
                return mat_lay_i.get_permittivity_ani(wavelength)*Id_3
        else:
            eps_tens_list = list()
            Id_3 = np.eye(3 , 3)
            for i in self.layer_type:
                mat_lay_i = self.materials[i]
                if mat_lay_i.type != "Anisotropic":
                    eps_tens_list.append(mat_lay_i.get_permittivity(wavelength)*Id_3)
                else :
                    eps_tens_list.append(mat_lay_i.get_permittivity_ani(wavelength)*Id_3)

        return eps_tens_list


    def rotate_permittivity_tensor(self, eps, ani_rot_angle, ani_rot_axis):
        """
            Rotates the permittivity tensor of a material layer
        """
        eps_R = rotate_permittivity(eps, ani_rot_angle, ani_rot_axis)
        return eps_R



    def rotate_permittivity_tensor_list(self, eps_tens_list, ani_rot_angle, ani_rot_axis):#AV_Added#
        """ Return the list of rotated permittivities tensors from permittivity_tensor_list. Each tensor is rotated about the corresponding angles and axis in the list ani_rot_angle and ani_rot_axis.

         Args:
         eps_tens_list : row array of 3*3 array
         ani_rot_angle : list of int or float
         ani_rot_axis : list of row array of length 3 or string 'x', 'y' or 'z'
        """
        i=0
        new_eps_tens_list = copy.deepcopy(eps_tens_list)
        for eps in new_eps_tens_list:
            eps_R = rotate_permittivity(eps, ani_rot_angle,ani_rot_axis)
            new_eps_tens_list[i] = eps_R
            i=i+1
        return new_eps_tens_list


    def plot_stack(self, wavelength=None, lim_eps_colors=[1.5, 4], precision=3):
        """plot layerstack

        evaluate materials at given wavelength for labels
        """
        if wavelength is None:
            wavelength=500.0
        elif(self.unit != "nm"):
            wavelength = conv_to_nm(wavelength, self.unit)

        _mats = []
        _mats_names = []
        _index_diff = []
        for i, mat in enumerate(np.array(self.materials)[self.layer_type]):
            if hasattr(mat, 'name') and len(mat.name)>0:
                # Named materials
                _mats_names.append(mat.name)
                _index_diff.append(i)
            # Simply using the permittivity at a given wavelength (default is 500 nm)
            _mats.append(mat.get_permittivity(wavelength))
        _index_diff = np.array(_index_diff)

        _thick = self.thickness

        ## give sub- and superstrate a finite thickness
        if _thick[0] == 0: _thick[0] = 50
        if _thick[-1] == 0: _thick[-1] = 50


        ## define colors for layers of different ref.indices
        if any(isinstance(x, str) for x in _mats):
            colors = ['.3'] + [f'C{i}'for i in range(len(_thick)-2)] + ['.3']
        else:
            cmap = plt.cm.jet
            colors = ['.3'] + [cmap((n.real-lim_eps_colors[0])/lim_eps_colors[1])
                                       for n in _mats[1:-1]] + ['.3']

        for i, di in enumerate(_thick):
            d0 = np.sum(_thick[:i])
            n = _mats[i]

            if type(n) is not str:
                if abs(np.imag(n)) < 1e-7:
                    n = np.real(n)
                n = str(np.round(n, 2))

            if i < len(_thick)-1:
                plt.axhline(d0+di, color='k', lw=1)
            plt.axhspan(d0, d0+di, color=colors[i], alpha=0.5)

            spacing = ""
            if len(_thick) > 12:
                spacing = " " #* np.random.randint(50)
            if i not in _index_diff:
                text = f'{spacing}eps={n}'
            else:
                n =  _mats_names[np.where(_index_diff==i)[0][0]]
                n = [float(s) for s in re.findall(r"-?\d+\.?\d*", n)][0]
                text = f'{spacing}mat={np.round(n, precision)}'

            if len(_thick)-1 > i >= 1:
                plt.text(0.05, d0+di/2, text, ha='left', va='center',fontsize=8)
                plt.text(0.95, d0+di/2, f'd={int(np.round(di))}nm{spacing}', ha='right', va='center',fontsize=8)
            else:
                plt.text(0.1, d0+di/2, text, ha='left', va='center',fontsize=8)
        plt.title(f'permittivities at wavelength = {wavelength}nm')
        plt.ylabel('D (nm)')
        plt.xticks([])
        plt.gca().invert_yaxis()
        plt.show()


class Beam:
    """ An object of the class contains all the parameters defining an incident
    beam. At initialization, a few messages will be displayed to inform the
    user.

    Args:
        wavelength (float): Wavelength in vacuum in nanometers
        incidence (float): Incidence angle in radians
        polarization (int) : '0' for TE polarization, TM otherwise
        waist (float): waist of the incident beam along the $x$ direction

    """

    def __init__(self, wavelength, incidence, polarization, horizontal_waist, unit="nm"):

        if (unit != "nm"):
            wavelength = conv_to_nm(wavelength, unit)
            horizontal_waist = conv_to_nm(horizontal_waist, unit)

        self.wavelength = wavelength
        self.incidence = incidence
        tmp = incidence * 180 / np.pi
        print("Incidence in degrees:", tmp)
        self.polarization = polarization
        if (polarization == 0):
            print("E//, TE, s polarization")
        else:
            print("H//, TM, p polarization")
        self.waist = horizontal_waist


class Window:
    """An object containing all the parameters defining the spatial domain
    which is represented.

    Args:
        width (float): width of the spatial domain (in nm)
        beam_relative_position (float): relative position of the source
        horizontal_pixel_size (float): size in nm of a pixel, horizontally
        vertical_pixel_size (float): size in nm of a pixel, vertically

    The number of pixel for each layer will be computed later, but the number of
    pixel horizontally is computed and stored in nx.

    The position of the center of the beam is automatically deduced from
    the relative position: 0 means complete left of the domain, 1 complete
    right and 0.5 in the middle of the domaine, of course.
    """

    def __init__(self, width, beam_relative_position, horizontal_pixel_size,
                 vertical_pixel_size, unit="nm"):

        if (unit != "nm"):
            width = conv_to_nm(width, unit)
            horizontal_pixel_size = conv_to_nm(horizontal_pixel_size, unit)
            vertical_pixel_size = conv_to_nm(vertical_pixel_size, unit)
        self.width = width
        self.C = beam_relative_position
        self.ny = 0
        self.px = float(horizontal_pixel_size)
        self.py = float(vertical_pixel_size)
        self.nx = int(np.floor(width / self.px))
        print("Pixels horizontally:", self.nx)



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
                self.material_list = [mat[0], mat[0], mat[1]]
                self.material_x = mat[0]
                self.material_y = mat[0]
                self.material_z = mat[1]
            elif (len(mat) == 3):
                # Biaxial, three values given,
                self.material_list = [mat[0], mat[1], mat[2]]
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
            if issubclass(material.__class__,RefractiveIndexMaterial):
                try:
                    k = material.get_extinction_coefficient(wavelength)
                    epsilon_medium.append(material.get_epsilon(wavelength)) # Here we use the fact that "get_epsilon(wl)" return an error if k is not given in the Ref Ind dataB to go in the except where we deal with the real index case. 
                    print('k =',k )
                    print('epsilon_medium =' ,material.get_epsilon(wavelength)) 
                except:                                                              # If k exist we use get_epsilon(wl) 
                    n = material.get_refractive_index(wavelength)
                    epsilon_medium.append(n**2)
                    print('n =',n)
            else:
                # Was directly given
                epsilon_medium.append(complex(material))
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


# TODO: this stays here for the moment, but should be removed evenually
authorized = {"permittivity_glass":None}
