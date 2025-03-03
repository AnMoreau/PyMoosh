"""
This file contains all functions for anisotropic material computations
"""

import numpy as np
import copy
from numpy import linalg as la_np
from PyMoosh.classes import Material, Structure, conv_to_nm
from refractiveindex import RefractiveIndexMaterial


def rotate_permittivity(eps, angle_rad, axis="z"):
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
        if axis == "x":
            axis = x_u
        elif axis == "y":
            axis = y_u
        elif axis == "z":
            axis = z_u
        else:
            raise Exception("Invalid rotation axis.")
    if la_np.norm(axis) == 0:  # if axis has zero norm then necessarily: axis=[0,0,0]
        raise Exception("Invalid axis. Axis can not be (0, 0, 0).")
    if np.array(axis).shape != x_u.shape:
        raise Exception("axis as to be a one-dimensional Numpy array of lenght 3")

    # Rotation matrix
    # theta_rad = theta_rad % (2 * np.pi)
    axis = axis / la_np.norm(axis)  # axis normalisation
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    costheta = np.cos(angle_rad)
    sintheta = np.sin(angle_rad)
    R = np.array(
        [
            [
                costheta + (ux**2) * (1 - costheta),
                ux * uy * (1 - costheta) - uz * sintheta,
                ux * uz * (1 - costheta) + uy * sintheta,
            ],
            [
                uy * ux * (1 - costheta) + uz * sintheta,
                costheta + (uy**2) * (1 - costheta),
                uy * uz * (1 - costheta) - ux * sintheta,
            ],
            [
                uz * ux * (1 - costheta) - uy * sintheta,
                uz * uy * (1 - costheta) + ux * sintheta,
                costheta + (uz**2) * (1 - costheta),
            ],
        ]
    )
    # Rotate permittivity tensor
    eps_rot = la_np.multi_dot((R, eps, R.transpose()))
    return eps_rot


class AniStructure(Structure):
    """
    Specific Structure class for multilayer structures containing
    Anisotropic material

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

    def __init__(
        self,
        materials,
        layer_type,
        thickness,
        ani_rot_angle=None,
        ani_rot_axis=None,
        verbose=True,
        unit="nm",
        si_units=False,
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
                # Checks if the material is already instanciated
                # NOTE 1: all Anisotropic materials should be instanciated
                # NOTE 2: should not be an non local material
                materials_final.append(mat)
                if verbose:
                    print("Material:", mat.__class__.__name__)
            else:
                new_mat = Material(mat, verbose=verbose)
                materials_final.append(new_mat)
        self.materials = materials_final
        self.layer_type = layer_type
        self.thickness = thickness

        if (
            ani_rot_angle == None
        ):  # Setting all the angles to 0 if nothing has been specified by the user.
            self.ani_rot_angle = [0] * np.size(layer_type)

        else:
            self.ani_rot_angle = ani_rot_angle  # ani_rot_angle is a list of angle in radian (float or int). One for each layer, setting isotropic layers to the default angle = 0

        for ang in self.ani_rot_angle:  # Checking format.
            if not (isinstance(ang, float) or isinstance(ang, int)):
                raise Exception("angle have to be a float or a int")

        if (
            ani_rot_axis == None
        ):  # Setting all the axis to 'z' if nothing has been specified by the user.
            self.ani_rot_axis = ["z"] * np.size(layer_type)
        else:
            self.ani_rot_axis = ani_rot_axis  # ani_rot-axis is a list of axis reprensented as a row array of length 3
            # or as the string ``'x'``, ``'y'`` or ``'z'``. One for each layer in the stack, setting isotropic layers to the default axis 'z'

        for ax in self.ani_rot_axis:  # Checking format.
            if not (isinstance(ax, str)) and np.shape(ax) != np.shape([0, 0, 0]):
                raise Exception(
                    "axis have to be a string ``'x'``, ``'y'`` or ``'z'` or a row array of length 3"
                )

        # Checking if the first and last layers are isotrop (Superstrate and Substrate are halfspaces respectively
        # representing the medium of the incoming and outgoing light of the multi-layer stack)
        if (
            materials_final[layer_type[0]].type == "Anisotropic"
            or materials_final[layer_type[-1]].type == "Anisotropic"
        ):
            raise Exception(
                "Superstrate's and Substrate's material have to be isotropic !"
            )

    def permittivity_tensor_list(self, wavelength, layer=None):  # AV_Added#
        """Return the permittivity tensor of each material considered in the structure as a row array of 3*3 array. Both isotropic and anisotropic materials are supported.

        Args:
        wavelength (float): the working wavelength (in nanometers)
        """
        if layer is not None:
            Id_3 = np.eye(3, 3)
            mat_lay_i = self.materials[self.layer_type[layer]]
            if mat_lay_i.type != "Anisotropic":
                return mat_lay_i.get_permittivity(wavelength) * Id_3
            else:
                return mat_lay_i.get_permittivity_ani(wavelength) * Id_3
        else:
            eps_tens_list = list()
            Id_3 = np.eye(3, 3)
            for i in self.layer_type:
                mat_lay_i = self.materials[i]
                if mat_lay_i.type != "Anisotropic":
                    eps_tens_list.append(mat_lay_i.get_permittivity(wavelength) * Id_3)
                else:
                    eps_tens_list.append(
                        mat_lay_i.get_permittivity_ani(wavelength) * Id_3
                    )

        return eps_tens_list

    def rotate_permittivity_tensor(self, eps, ani_rot_angle, ani_rot_axis):
        """
        Rotates the permittivity tensor of a material layer
        """
        eps_R = rotate_permittivity(eps, ani_rot_angle, ani_rot_axis)
        return eps_R

    def rotate_permittivity_tensor_list(
        self, eps_tens_list, ani_rot_angle, ani_rot_axis
    ):  # AV_Added#
        """Return the list of rotated permittivities tensors from permittivity_tensor_list. Each tensor is rotated about the corresponding angles and axis in the list ani_rot_angle and ani_rot_axis.

        Args:
        eps_tens_list : row array of 3*3 array
        ani_rot_angle : list of int or float
        ani_rot_axis : list of row array of length 3 or string 'x', 'y' or 'z'
        """
        i = 0
        new_eps_tens_list = copy.deepcopy(eps_tens_list)
        for eps in new_eps_tens_list:
            eps_R = rotate_permittivity(eps, ani_rot_angle, ani_rot_axis)
            new_eps_tens_list[i] = eps_R
            i = i + 1
        return new_eps_tens_list


class AniMaterial(Material):
    """
    Anisotropic material class

    Can be instanciated in two ways:
    - From RefractiveIndex -> specialType should be ANI_RII
                              mat should be shelf, book, page
    - Given by hand -> special Type should be ANI
                       mat should be a list of 2 (uniaxial) or 3 (biaxial) permittivities
    - Given by hand and dispersive -> special Type should be Model_ANI
                       mat should be a list of 2 (uniaxial) or 3 (biaxial) [permittivity functions, params]
    """

    def __init__(self, mat, specialType="ANI", verbose=False):
        self.type = "Anisotropic"

        if specialType == "ANI":
            # User defined Anisotropic material
            if len(mat) < 2 or len(mat) > 3:
                print(
                    f"Warning: Anisotropic material is expected to be a list of 2 or 3 index values, but {len(mat)} were given."
                )
            self.specialType = specialType
            if len(mat) == 2:
                # Uniaxial, only two values given, no and ne
                self.material_list = [mat[0], mat[0], mat[1]]
            elif len(mat) == 3:
                # Biaxial, three values given,
                self.material_list = [mat[0], mat[1], mat[2]]

            self.name = "Anisotropic material" + str(mat)
            if verbose:
                print("Anisotropic material of indices ", str(mat))

        if specialType == "Model_ANI":
            # User defined Anisotropic material defined with functions
            if len(mat) < 2 or len(mat) > 3:
                print(
                    f"Warning: Anisotropic material is expected to be a list of 2 or 3 index values, but {len(mat)} were given."
                )
            self.specialType = specialType
            if len(mat) == 2:
                # Uniaxial, only two values given, no and ne
                self.material_list = [mat[0], mat[0], mat[1]]
            elif len(mat) == 3:
                # Biaxial, three values given,
                self.material_list = [mat[0], mat[1], mat[2]]

            self.name = "Anisotropic dispersive material" + str(mat)
            if verbose:
                print("Anisotropic dispersive material of functions ", str(mat))

        elif specialType == "ANI_RII":
            # Anisotropic material from the refractive index database
            if len(mat) != 3:
                print(
                    f"Warning: Anisotropic material from Refractiveindex.info is expected to be a list of 3 values, but {len(mat)} were given."
                )
            self.specialType = specialType
            shelf, book, page = mat[0], mat[1], mat[2]
            self.path = "shelf: {}, book: {}, page: {}".format(
                shelf, book, page
            )  # not necessary ?
            material_list = wrapper_anisotropy(
                shelf, book, page
            )  # A list of three materials
            self.material_list = material_list
            self.name = "Anisotropic material from Refractiveindex.info: " + str(mat)
            if verbose:
                print("Material from Refractiveindex Database")
            if len(mat) != 3:
                print(
                    f"Warning: Material from RefractiveIndex Database should have 3 values (shelf, book, page), but {len(mat)} were given."
                )

    def get_permittivity_ani(self, wavelength):
        """
        Get the permittivities of the anisotropic material
        Either uses the built in function from RefractiveIndex
        or the value given by the user
        """
        epsilon_medium = []
        for material in self.material_list:
            if issubclass(material.__class__, RefractiveIndexMaterial):
                try:
                    k = material.get_extinction_coefficient(wavelength)
                    epsilon_medium.append(
                        material.get_epsilon(wavelength)
                    )  # Here we use the fact that "get_epsilon(wl)" return an error if k is not given in the Ref Ind dataB to go in the except where we deal with the real index case.
                    print("k =", k)
                    print("epsilon_medium =", material.get_epsilon(wavelength))
                except:  # If k exist we use get_epsilon(wl)
                    n = material.get_refractive_index(wavelength)
                    epsilon_medium.append(n**2)
                    print("n =", n)
            elif self.specialType == "Model_ANI":
                function = material[0]
                params = material[1:]
                epsilon_medium.append(complex(function(wavelength, *params)))
            else:
                # Was directly given
                epsilon_medium.append(complex(material))
        return epsilon_medium


def wrapper_anisotropy(shelf, book, page):
    """
    Helper function to find the correct RefractiveIndex pages:
    2 main cases:
        - uniaxial material, only 2 values stored as
        ordinary (-o) and extraordinary (-e) values
        - biaxial material, 3 different values, stored as
        -alpha, -beta, and -gamma
    """
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
            page_a, page_b, page_c = (
                page,
                page.replace("-alpha", "-beta"),
                page.replace("-alpha", "-gamma"),
            )
        elif page.endswith("-beta"):
            page_a, page_b, page_c = (
                page.replace("-beta", "-alpha"),
                page,
                page.replace("-beta", "-gamma"),
            )
        elif page.endswith("-gamma"):
            page_a, page_b, page_c = (
                page.replace("-gamma", "-alpha"),
                page.replace("-gamma", "-beta"),
                page,
            )

        # create ordinary and extraordinary object.
        material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
        material_beta = RefractiveIndexMaterial(shelf, book, page_b)
        material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
        return [material_alpha, material_beta, material_gamma]

    else:
        # No extensions were given, so we have to try both
        # uniaxial and biaxial cases
        try:  # uniaxial
            page_e, page_o = page + "-e", page + "-o"
            material_o = RefractiveIndexMaterial(shelf, book, page_o)
            material_e = RefractiveIndexMaterial(shelf, book, page_e)
            return [material_o, material_o, material_e]
        except:
            try:  # biaxial
                page_a, page_b, page_c = (
                    page + "-alpha",
                    page + "-beta",
                    page + "-gamma",
                )
                print(page_a)
                material_alpha = RefractiveIndexMaterial(shelf, book, page_a)
                print(material_alpha)
                material_beta = RefractiveIndexMaterial(shelf, book, page_b)
                print(material_beta)
                material_gamma = RefractiveIndexMaterial(shelf, book, page_c)
                print(material_gamma)
                return [material_alpha, material_beta, material_gamma]
            except:
                print(
                    f'Warning: Given material is not known to be anisotropic in the Refractiveindex.info database. You should try to remove "ANI" keyword in material definition or to spellcheck the given path.'
                )


def calc_cp(Sx, Sy):
    """
    Compute poynting in the x direction in the x,y plane
    """
    deno = np.abs(Sx) ** 2 + np.abs(Sy) ** 2
    if deno == 0:
        return 0
    else:
        return np.abs(Sx) ** 2 / deno


def Halfspace_method(struct, layer_number, wl, theta_entry):  # AV_Added#
    """
    This function calculates the HalfSpace's eigenvectors and eigenvalues analytically for a given layer and returns them sorted.

    Args:
    struct (class): description of the multi-mayered
    layer_number (int): position in the stack of the studied layer
    wl (float): the working wavelength (in nanometers)
    theta_entry (float): angle of the incident light

    return: p_sorted 4x4 array of eigenvectors whose columns are the same as p_unsorted's but sorted
    and q_sorted 4x4 diagonal matrix of eigenvalues whose columns are the same as q_unsorted's but sorted
    """
    k_0 = 2 * np.pi / wl
    eps_entry = struct.permittivity_tensor_list(wl)[layer_number]
    n = np.sqrt(eps_entry[0, 0])
    Kx = n * np.sin(theta_entry)

    # Getting the angles necessary for the eigen vectors of the halfspac
    sin_phi = Kx / n
    cos_phi = np.sqrt(1 - sin_phi**2 + 0j)
    # q_sorted = [n * cos_phi, n * cos_phi, -n * cos_phi, -n * cos_phi]
    p_sorted_mat = np.array(
        [
            [cos_phi, 0, cos_phi, 0],
            [n, 0, -n, 0],
            [0, 1, 0, 1],
            [0, n * cos_phi, 0, -n * cos_phi],
        ]
    )
    Q = np.array([1, 1, 1, 1], dtype=complex)  # ref: PyLlama p14

    return p_sorted_mat, Q


def Berreman_method(struct, layer_number, wl, theta_entry):  # AV_Added#
    """
    This function computes Berreman's matrix D for a given layer "layer_number" in the stack and its associated eigenvalues q and associated eigenvectors for a given layer "layer_number" in the stack .
    Then P (interface matrix) and Q (propagation matirix) are computed layer in the stack

    Args:
    struct (class): description of the multi-mayered
    layer_number (int): position in the stack of the studied layer
    wl (float): the working wavelength (in nanometers)
    theta_entry (float): angle of the incident light

    return: Delta, P and Q matrices as 4x4 Numpy array

    """
    k_0 = 2 * np.pi / wl
    eps_entry = struct.permittivity_tensor_list(wl, layer=0)
    n_entry = np.sqrt(eps_entry[0, 0])
    Kx = n_entry * np.sin(theta_entry)

    eps = struct.permittivity_tensor_list(wl, layer=layer_number)

    eps_R = struct.rotate_permittivity_tensor(
        eps, struct.ani_rot_angle[layer_number], struct.ani_rot_axis[layer_number]
    )

    # Delta matrix  (i.e Berreman matrix)

    eps_xx = eps_R[0, 0]
    eps_xy = eps_R[0, 1]
    eps_yx = eps_R[1, 0]
    eps_yy = eps_R[1, 1]
    eps_xz = eps_R[0, 2]
    eps_zx = eps_R[2, 0]
    eps_zz = eps_R[2, 2]
    eps_zy = eps_R[2, 1]
    eps_yz = eps_R[1, 2]
    Delta = np.array(
        [
            [-Kx * eps_zx / eps_zz, 1 - Kx**2 / eps_zz, -Kx * eps_zy / eps_zz, 0],
            [
                eps_xx - (eps_xz * eps_zx) / eps_zz,
                -Kx * eps_xz / eps_zz,
                eps_xy - (eps_xz * eps_zy) / eps_zz,
                0,
            ],
            [0, 0, 0, 1],
            [
                eps_yx - (eps_yz * eps_zx) / eps_zz,
                -Kx * eps_yz / eps_zz,
                -(Kx**2) + eps_yy - (eps_yz * eps_zy) / eps_zz,
                0,
            ],
        ]
    )

    q, P = la_np.eig(Delta)
    # q is the row vector containing unsorted Delta's eigenvalues and
    # # P is the array which column P[:,i] is the eigenvector corresponding to the eigenvalues q[i].
    jk0h = 1j * k_0 * struct.thickness[layer_number]
    Q = np.exp(jk0h * q)
    # Propagation matrix

    # Sorting wavevectors and eigenvalues to construct "forward" and "backward" matrices
    thr = 1e-7  # birefringence detection threshold
    id_refl = []
    id_trans = []
    Ex_list = []
    Ey_list = []
    Sx_list = []
    Sy_list = []
    Sz_list = []
    for k in range(0, 4, 1):
        # Computing fields => ref. PyLlama equation (7) p9) :
        Ex, Ey, Hx, Hy = P[0, k], P[2, k], -P[3, k], P[1, k]
        Ez = -(eps_zx / eps_zz) * Ex - (eps_zy / eps_zz) * Ey - (Kx / eps_zz) * Hy
        Hz = Kx * Ey

        # Computing Poynting vector :
        Sx = Ey * Hz - Ez * Hy
        Sy = Ez * Hx - Ex * Hz
        Sz = Ex * Hy - Ey * Hx
        Ex_list.append(Ex)
        Ey_list.append(Ey)
        Sx_list.append(Sx)
        Sy_list.append(Sy)
        Sz_list.append(Sz)
        # poynting = [Sx, Sy, Sz]
        if np.isreal(Sz):
            # We sort first along real Poynting values
            test_variable = np.real(Sz)
        else:
            # and then we sort complex Poynting values wrt their imaginary parts
            test_variable = np.imag(Sz)
        if test_variable > 0:
            # There'll always be one positive and one negative value.
            # The positive is going down (like transmission)
            id_trans.append(k)
        else:
            # The negative is going up (like reflection)
            id_refl.append(k)
    Cp0 = calc_cp(Sx_list[id_trans[0]], Sy_list[id_trans[0]])
    Cp1 = calc_cp(Sx_list[id_trans[1]], Sy_list[id_trans[1]])
    if np.abs(Cp0 - Cp1) > thr:
        # It is birefringent, we sort according to Poynting vectors
        if Cp1 < Cp0:
            id_trans = [id_trans[1], id_trans[0]]
        Cp0 = calc_cp(Sx_list[id_refl[0]], Sy_list[id_refl[0]])
        Cp1 = calc_cp(Sx_list[id_refl[1]], Sy_list[id_refl[1]])
        if Cp1 < Cp0:
            id_refl = [id_refl[1], id_refl[0]]
    else:
        # It is not birefringent, we sort according to Electric fields
        Cp0 = calc_cp(Ex_list[id_trans[0]], Ey_list[id_trans[0]])
        Cp1 = calc_cp(Ex_list[id_trans[1]], Ey_list[id_trans[1]])
        if (Cp1 - Cp0) < thr:
            id_trans = [id_trans[1], id_trans[0]]
        Cp0 = calc_cp(Ex_list[id_refl[0]], Ey_list[id_refl[0]])
        Cp1 = calc_cp(Ex_list[id_refl[1]], Ey_list[id_refl[1]])
        if (Cp1 - Cp0) < thr:
            id_refl = [id_refl[1], id_refl[0]]

    # Sort p and q with the newly computed order
    order = [id_trans[1], id_trans[0], id_refl[1], id_refl[0]]

    Q = np.array([Q[order[0]], Q[order[1]], Q[order[2]], Q[order[3]]])

    P = np.stack(
        (
            P[:, order[0]].transpose(),
            P[:, order[1]].transpose(),
            P[:, order[2]].transpose(),
            P[:, order[3]].transpose(),
        ),
        axis=1,
    )

    return Delta, P, Q


def build_scattering_matrix_to_next(P_a, Q_a, P_b):
    """
    This function constructs the scattering matrix S_{ab} between two successive layers a and
    b by taking into acount the following phenomena:
    - the propagation through the first layer with the propagation matrix Q_a of layer_a
    - the transition from the first layer (layer_a's matrix P_a) and the second layer (layer_b's matrix P_b)

    Args:
        transition matrix : P_a
        propagation matrix : Q_a
        transition matrix : P_b

    :return: partial scattering matrix from layer a to layer b, a 4x4 Numpy array
    """

    Q_forward = np.diag([Q_a[0], Q_a[1], 1, 1])

    Q_backward = np.diag([1, 1, Q_a[2], Q_a[3]])

    P_out = np.block([[P_a[:, :2], -P_b[:, 2:]]])

    P_in = np.block([P_b[:, :2], -P_a[:, 2:]])
    Sab = la_np.multi_dot((la_np.inv(Q_backward), la_np.inv(P_in), P_out, Q_forward))
    return Sab


def combine_scattering_matrices(S_ab, S_bc):
    """
    This function constructs the scattering matrix between three successive layers a, b and c by combining
    the scattering matrices S_{ab} from layer a to layer b and S_{bc} from layer b to layer c.

    :param ndarray S_ab: the scattering matrix from layer a to layer b, a 4x4 Numpy array
    :param ndarray S_bc: the scattering matrix from layer b to layer c, a 4x4 Numpy array
    :return: partial scattering matrix from layer a to layer c, a 4x4 Numpy array
    """
    S_ab00 = S_ab[:2, :2]
    S_ab01 = S_ab[:2, 2:]
    S_ab10 = S_ab[2:, :2]
    S_ab11 = S_ab[2:, 2:]

    S_bc00 = S_bc[:2, :2]
    S_bc01 = S_bc[:2, 2:]
    S_bc10 = S_bc[2:, :2]
    S_bc11 = S_bc[2:, 2:]

    C = la_np.inv(np.identity(2) - np.dot(S_ab01, S_bc10))
    S_ac00 = la_np.multi_dot((S_bc00, C, S_ab00))
    S_ac01 = S_bc01 + la_np.multi_dot((S_bc00, C, S_ab01, S_bc11))
    S_ac10 = S_ab10 + la_np.multi_dot((S_ab11, S_bc10, C, S_ab00))
    S_ac11 = la_np.multi_dot(
        (S_ab11, (np.identity(2) + la_np.multi_dot((S_bc10, C, S_ab01))), S_bc11)
    )

    S_ac = np.block([[S_ac00, S_ac01], [S_ac10, S_ac11]])
    return S_ac


def coefficients_ani(structure, wl, theta_inc):
    """
    This function returns the four reflection and transmission coefficients of the all structure from its global scattering matrix.
    To get P and Q for each layer, calculation of the eigenvalues and eigenvectors are done analytically for the superstrate and the substrate with Halfspace_method()
    In every other layer we computes these e.val and e.vect with Berreman_method() (containing the sorting algorithm).

    Args:
    struct (class): description of the multi-mayered
    wl (float): the working wavelength (in nanometers)
    theta_inc (float): angle of the incident light

    return: t_pp, t_ps, t_sp, t_ss, r_pp, r_ps, r_sp and r_ss

    """
    # step 1: create all P, Q matrices (interface and propagation resp.)
    P_list = []
    Q_list = []

    # Upper Halfspace (superstrate)
    P, Q = Halfspace_method(structure, 0, wl, theta_inc)
    P_list.append(P)
    Q_list.append(Q)
    for layer_number in range(1, len(structure.layer_type) - 1):

        D, P, Q = Berreman_method(structure, layer_number, wl, theta_inc)
        P_list.append(P)
        Q_list.append(Q)

    # Lower Halfspace (substrate)
    P, Q = Halfspace_method(structure, len(structure.layer_type) - 1, wl, theta_inc)
    P_list.append(P)
    Q_list.append(Q)

    # step 2 : create S matrices
    S_last_period = np.identity(4)
    for lay in range(len(structure.layer_type) - 2, -1, -1):
        Sab = build_scattering_matrix_to_next(P_list[lay], Q_list[lay], P_list[lay + 1])
        S_last_period = combine_scattering_matrices(Sab, S_last_period)

    # step 3 : extract r and t coefficients (r_kj <=> reflection coeff. of a k polarized incident wave reflected in a j polarized wave)
    # (t_kj <=> transmison coeff. of a k polarized incident wave transmitted in a j polarized wave)

    t_pp = S_last_period[0, 0]
    t_ps = S_last_period[0, 1]
    t_sp = S_last_period[1, 0]
    t_ss = S_last_period[1, 1]

    r_pp = S_last_period[2, 0]
    r_ps = S_last_period[2, 1]
    r_sp = S_last_period[3, 0]
    r_ss = S_last_period[3, 1]

    return t_pp, t_ps, t_sp, t_ss, r_pp, r_ps, r_sp, r_ss
