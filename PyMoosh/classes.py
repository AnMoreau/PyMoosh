"""
This file contains all the basic structures necessary for PyMoosh to run
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.special import wofz
import json
from refractiveindex import RefractiveIndexMaterial


def conv_to_nm(length, unit):
    """Converts a length from "unit"to nm, because everything has been coded
    in nm...
    """
    if unit == "m":
        return np.array(length) * 1e9
    elif unit == "um":
        return np.array(length) * 1e3
    elif unit == "mm":
        return np.array(length) * 1e6
    elif unit == "pm":
        return np.array(length) * 1e-3
    elif unit == "nm":
        # Just in case we get here but didn't need to
        return np.array(length)
    else:
        print("Please provide lengths in m, mm, um, pm or nm")


class Structure:
    """Each instance of Structure describes a multilayer completely.
    This includes the materials the multilayer is made of and the
    thickness of each layer.

    Args:
        materials (list) : a list of material definition
        layer_type (list) : how the different materials are stacked
        thickness (list) : thickness of each layer in nm
        units (str) : the length unit used for thickness

    Materials can be defined in the list :materials:
    -by giving their permittivity as a complex number for non dispersive materials
    -by giving a list of two complex numbers. The first is the permittivity and the
    second one the permeability. It is considered non dispersive.
    -by giving its name for dispersive materials that are in the database.
    -by giving a custom permittivity function, taking as an argument the
    wavelength in nanometer.

    Other material types must be implemented beforehand, see the :Material: class.


    Example: [1.,'Si','Au'] means we will use a material with a permittivity
    of 1. (air), silicon and gold.

    Each material can then be refered to by its position in the list :materials:
    The list layer_type is used to describe how the different materials are
    placed in the structure, using the place of each material in the list
    :materials:

    Example: [0,1,2,0] describes a superstrate made of air (permittivity 1.),
    on top of Si, on top of gold and a substrate made of air.

    The thickness of each layer is given in the :thickness: list, in nanometers
    by default. The thickness of the superstrate is assumed to be zero by most
    of the routines (like :coefficient:, :absorption:) so that the first
    interface is considered as the phase reference. The reflection coefficient
    of the structure will thus never be impacted by the thickness of the first
    layer.For other routines (like :field:), this thickness define the part
    of the superstrate (or substrate) that must be represented on the figure.

    Example with the previous definitions:
    [0,200,300,500]  refers to an infinite air superstrate, a 200 nm thick silicon layer,
    a 300 nm thick gold layer and an infinite air substrate.
    The thicknessees of superstrates and substrates are here for visualisation purposes.
    """

    def __init__(
        self, materials, layer_type, thickness, verbose=True, unit="nm", si_units=False
    ):

        if unit != "nm":
            thickness = conv_to_nm(thickness, unit)
            if not (si_units):
                print(
                    "I can see you are using another unit than nanometers, ",
                    "please make sure you keep using that unit everywhere.",
                    " To suppress this message, add the keyword argument si_units=True when you call Structure",
                )

        self.unit = unit

        materials_final = list()
        if verbose:
            print("List of materials:")
        for mat in materials:
            if issubclass(mat.__class__, Material):
                materials_final.append(mat)
                if verbose:
                    print("Material:", mat.type)
            else:
                new_mat = Material(mat, verbose=verbose)
                materials_final.append(new_mat)
        self.materials = materials_final
        self.layer_type = layer_type
        self.thickness = thickness

    def __str__(self):
        materials = [str(self.materials[i]) for i in range(len(self.materials))]
        s = f"materials: {materials}\nlayers: {self.layer_type}\nthicknesses: {self.thickness}"
        return s

    def polarizability(self, wavelength):
        """Computes the actual permittivity and permeability of each material considered in
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

    def plot_stack(self, wavelength=None, lim_eps_colors=[1.5, 4], precision=3):
        """plot layerstack

        evaluate materials at given wavelength for labels
        """
        if wavelength is None:
            wavelength = 500.0
        elif self.unit != "nm":
            wavelength = conv_to_nm(wavelength, self.unit)

        _mats = []
        _mats_names = []
        _index_diff = []
        for i, mat in enumerate(np.array(self.materials)[self.layer_type]):
            if hasattr(mat, "name") and len(mat.name) > 0:
                # Named materials
                _mats_names.append(mat.name)
                _index_diff.append(i)
            # Simply using the permittivity at a given wavelength (default is 500 nm)
            _mats.append(mat.get_permittivity(wavelength))
        _index_diff = np.array(_index_diff)

        _thick = self.thickness

        ## give sub- and superstrate a finite thickness
        if _thick[0] == 0:
            _thick[0] = 50
        if _thick[-1] == 0:
            _thick[-1] = 50

        ## define colors for layers of different ref.indices
        if any(isinstance(x, str) for x in _mats):
            colors = [".3"] + [f"C{i}" for i in range(len(_thick) - 2)] + [".3"]
        else:
            cmap = plt.cm.jet
            colors = (
                [".3"]
                + [
                    cmap((n.real - lim_eps_colors[0]) / lim_eps_colors[1])
                    for n in _mats[1:-1]
                ]
                + [".3"]
            )

        for i, di in enumerate(_thick):
            d0 = np.sum(_thick[:i])
            n = _mats[i]

            if type(n) is not str:
                if abs(np.imag(n)) < 1e-7:
                    n = np.real(n)
                n = str(np.round(n, 2))

            if i < len(_thick) - 1:
                plt.axhline(d0 + di, color="k", lw=1)
            plt.axhspan(d0, d0 + di, color=colors[i], alpha=0.5)

            spacing = ""
            if len(_thick) > 12:
                spacing = " "  # * np.random.randint(50)
            # if i not in _index_diff:
            text = f"{spacing}eps={n}"
            # else:
            #     n =  _mats_names[np.where(_index_diff==i)[0][0]]
            #     n = [float(s) for s in re.findall(r"-?\d+\.?\d*", n)][0]
            #     text = f'{spacing}mat={np.round(n, precision)}'

            if len(_thick) - 1 > i >= 1:
                plt.text(0.05, d0 + di / 2, text, ha="left", va="center", fontsize=8)
                plt.text(
                    0.95,
                    d0 + di / 2,
                    f"d={int(np.round(di))}nm{spacing}",
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            else:
                plt.text(0.1, d0 + di / 2, text, ha="left", va="center", fontsize=8)
        plt.title(f"permittivities at wavelength = {wavelength}nm")
        plt.ylabel("D (nm)")
        plt.xticks([])
        plt.gca().invert_yaxis()
        plt.show()


class Beam:
    """
    An object of the class contains all the parameters defining an incident
    beam. At initialization, a few messages will be displayed to inform the
    user.

    Args:
        wavelength (float): Wavelength in vacuum in nanometers
        incidence (float): Incidence angle in radians
        polarization (int) : '0' for TE polarization, TM otherwise
        waist (float): waist of the incident beam along the $x$ direction

    """

    def __init__(
        self, wavelength, incidence, polarization, horizontal_waist, unit="nm"
    ):

        if unit != "nm":
            wavelength = conv_to_nm(wavelength, unit)
            horizontal_waist = conv_to_nm(horizontal_waist, unit)

        self.wavelength = wavelength
        self.incidence = incidence
        tmp = incidence * 180 / np.pi
        print("Incidence in degrees:", tmp)
        self.polarization = polarization
        if polarization == 0:
            print("E//, TE, s polarization")
        else:
            print("H//, TM, p polarization")
        self.waist = horizontal_waist


class Window:
    """
    An object containing all the parameters defining the spatial domain
    to be computed (considered periodic).

    Args:
        width (float): width of the spatial domain (in nm)
        beam_relative_position (float): relative position of the source
        horizontal_pixel_size (float): size in nm of a pixel, horizontally
        vertical_pixel_size (float): size in nm of a pixel, vertically

    The number of pixel for each layer will be computed later, but the number of
    pixel horizontally is computed and stored in nx.

    The position of the center of the beam is automatically deduced from
    the relative position: 0 means complete left of the domain, 1 complete
    right and 0.5 in the middle of the domaine.
    """

    def __init__(
        self,
        width,
        beam_relative_position,
        horizontal_pixel_size,
        vertical_pixel_size,
        unit="nm",
    ):

        if unit != "nm":
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


class Material:
    """
    Types of material (default):
          type                   / format of "mat" variable:

        - material               / Material object
        - CustomFunction         / function (wav)
        - simple_perm            / complex
        - magnetic               / list(complex, float) of size 2
        - Database               / string
              Database types can take many special types

        There are three special types:
        -> when importing from the Refractive Index Database
            Then the specialType variable should be set to "RII"
            - RefractiveIndexInfo    / list(shelf, book, page)
        -> when using a function with custom parameters
            Then the specialType variable should be set to "Model"
            - Model                  / list(function(wav, params), params)
        -> when using two functions with custom parameters for permittivity + permeability
            Then the specialType variable should be set to "ModelMu"
            - ModelMu                / [list(function(wav, params), params), list(function(wav, params), params)]

        And these materials have to be processed through the Material constructor first
        before being fed to Structure as Material objects

    """

    def __init__(self, mat, specialType="Default", verbose=False):

        if issubclass(mat.__class__, Material):
            # Has already been processed by this constructor previously
            if verbose:
                print("Preprocessed material:", mat.__name__)

        if specialType == "Default":
            # The default behaviors listed in the docstring
            self.specialType = specialType
            if mat.__class__.__name__ == "function":
                # Is a custom function that only takes the wavelength as a parameter
                self.type = "CustomFunction"
                self.permittivity_function = mat
                self.name = "CustomFunction: " + mat.__name__
                if verbose:
                    print(
                        "Custom dispersive material. Epsilon=",
                        mat.__name__,
                        "(wavelength in nm)",
                    )

            elif not hasattr(mat, "__iter__"):
                # no func / not iterable --> single value, convert to complex by default
                self.type = "simple_perm"
                self.name = "SimplePermittivity:" + str(mat)
                self.permittivity = complex(mat)
                if verbose:
                    print("Simple, non dispersive: epsilon=", self.permittivity)

            elif (
                isinstance(mat, list)
                and (isinstance(mat[0], float) or isinstance(mat[0], complex))
                and (isinstance(mat[1], float) or isinstance(mat[1], complex))
            ):
                # magnetic == [complex, complex]
                # iterable: if list or similar --> magnetic
                self.type = "magnetic"
                self.permittivity = mat[0]
                self.permeability = mat[1]
                self.name = (
                    "MagneticPermittivity:" + str(mat[0]) + "Permability:" + str(mat[1])
                )
                if verbose:
                    print("Magnetic, non dispersive: epsilon=", mat[0], " mu=", mat[1])
                if len(mat) > 2:
                    print(
                        f"Warning: Magnetic material should have 2 values (epsilon / mu), but {len(mat)} were given."
                    )

            elif isinstance(mat, str):
                # iterable: string --> database material from file in shipped database
                import pkgutil

                f = pkgutil.get_data(__name__, "data/material_data.json")
                f_str = f.decode(encoding="utf8")
                database = json.loads(f_str)
                if mat in database:
                    material_data = database[mat]
                    model = material_data["model"]

                    if model == "ExpData":
                        # Experimnental data to be interpolated
                        self.type = "ExpData"
                        self.name = "ExpData: " + str(mat)

                        wl = np.array(material_data["wavelength_list"])
                        epsilon = np.array(material_data["permittivities"])
                        if "permittivities_imag" in material_data:
                            epsilon = epsilon + 1j * np.array(
                                material_data["permittivities_imag"]
                            )

                        self.wavelength_list = np.array(wl, dtype=float)
                        self.permittivities = np.array(epsilon, dtype=complex)

                    elif model == "BrendelBormann":
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

                    else:
                        print(model, " not an existing model (yet).")
                        # sys.exit()

                    if verbose:
                        print("Database material:", self.name)
                else:
                    print(mat, "Unknown material in the database (for the moment)")
                    # print("Known materials:\n", existing_materials())
                    # sys.exit()

            else:
                print(
                    f"Warning: Given data is not in the right format for a 'Default' specialType. You should check the data format or specify a specialType. You can refer to the following table:"
                )
                print(self.__doc__)

        elif specialType == "RII":
            # Refractive index material
            if len(mat) != 3:
                print(
                    f"Warning: Material RefractiveIndex Database is expected to be a list of 3 values, but {len(mat)} were given."
                )
            self.type = "RefractiveIndexInfo"
            self.specialType = specialType
            self.name = "MaterialRefractiveIndexDatabase: " + str(mat)
            shelf, book, page = mat[0], mat[1], mat[2]
            material = RefractiveIndexMaterial(shelf, book, page)  # create object
            self.material = material
            if verbose:
                # print("Hello there ;)")
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(
                    f"Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given."
                )

        elif specialType == "Model":
            # A custom function that takes more parameters than simply the wavelength
            self.type = "Model"
            self.specialType = specialType
            self.permittivity_function = mat[0]
            self.params = [mat[i + 1] for i in range(len(mat) - 1)]
            self.name = "Customfunction: " + str(mat[0])

        elif specialType == "ModelMu":
            # Two custom functions that take more parameters than simply the wavelength
            self.type = "ModelMu"
            self.specialType = specialType
            eps = mat[0]
            self.permittivity_function = eps[0]
            self.eps_params = [eps[i + 1] for i in range(len(eps) - 1)]
            mu = mat[1]
            self.permeability_function = mu[0]
            self.mu_params = [mu[i + 1] for i in range(len(mu) - 1)]
            self.name = (
                "CustomfunctionMu: "
                + str(self.permittivity_function)
                + " "
                + str(self.permeability_function)
            )
            if verbose:
                print(
                    "CustomfunctionMu: "
                    + str(self.permittivity_function.__name__)
                    + " "
                    + str(self.permeability_function.__name__)
                )

        else:
            print(f"Warning: Unknown type : {specialType}")

    def __str__(self):
        return self.name

    def get_permittivity(self, wavelength):
        if self.type == "simple_perm":
            return self.permittivity

        elif self.type == "magnetic":
            return self.permittivity

        elif self.type == "CustomFunction":
            return self.permittivity_function(wavelength)

        elif self.type == "Model":
            return self.permittivity_function(wavelength, *self.params)

        elif self.type == "ModelMu":
            return self.permittivity_function(wavelength, *self.eps_params)

        elif self.type == "BrendelBormann":
            w = 6.62607015e-25 * 299792458 / 1.602176634e-19 / wavelength
            a = np.sqrt(w * (w + 1j * self.gamma))
            x = (a - self.omega) / (np.sqrt(2) * self.sigma)
            y = (a + self.omega) / (np.sqrt(2) * self.sigma)
            # Polarizability due to bound electrons
            chi_b = np.sum(
                1j
                * np.sqrt(np.pi)
                * self.f
                * self.omega_p**2
                / (2 * np.sqrt(2) * a * self.sigma)
                * (wofz(x) + wofz(y))
            )
            # Equivalent polarizability linked to free electrons (Drude model)
            chi_f = -self.omega_p**2 * self.f0 / (w * (w + 1j * self.Gamma0))
            epsilon = 1 + chi_f + chi_b
            return epsilon

        elif self.type == "ExpData":
            return np.interp(wavelength, self.wavelength_list, self.permittivities)

        elif self.type == "RefractiveIndexInfo":
            try:
                k = self.material.get_extinction_coefficient(wavelength)
                return self.material.get_epsilon(wavelength)
            except:
                n = self.material.get_refractive_index(wavelength)
                return n**2

    def get_permeability(self, wavelength, verbose=False):
        if self.type == "magnetic":
            return self.permeability
        if self.type == "ModelMu":
            return self.permeability_function(wavelength, *self.mu_params)
        elif self.type == "RefractiveIndexInfo":
            if verbose:
                print(
                    "Warning: Magnetic parameters from RefractiveIndex Database are not implemented. Default permeability is set to 1.0 ."
                )
            return 1.0
        return 1.0
