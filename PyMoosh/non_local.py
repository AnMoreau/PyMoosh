"""
This file contains all functions necessary to compute
the behavior of non-local (spatially dispersive) materials/structures
"""

import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from PyMoosh.core import conv_to_nm
from PyMoosh.classes import Material, Structure

# TODO: add other functionalities (field/absorption, etc.)


class NLStructure(Structure):
    """
    Specific function for multilayer structures containing Non Local materials
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
                # Checks if the material is already instanciated
                # NOTE 1: all NL materials should be instanciated
                # NOTE 2: should not be an anisotropic material
                materials_final.append(mat)
                if verbose:
                    print("Material:", mat.__class__.__name__)
            else:
                new_mat = Material(mat, verbose=verbose)
                materials_final.append(new_mat)
        self.materials = materials_final
        self.layer_type = layer_type
        self.thickness = thickness


class NLMaterial(Material):
    """
    A specific class, child of Material, to manage Non local materials

    From the old material function

    Non local materials
        - custom function based / function   and params         / 'NonLocal'       / 'NonLocalModel'

        All non local materials need: beta0, tau, omegap
        + all the parameters needed in their respective functions
        custom function must return: beta2, chi_b, chi_f, omega_p
    """

    def __init__(self, mat, verbose=False):
        self.specialType = "NonLocal"
        # If mat is a function then
        if callable(mat):
            self.type = "NonLocalModel"
            self.name = "NonLocalModel : " + str(mat)
            self.NL_function = mat
            self.params = []  # Pas de paramètres supplémentaires
            if verbose:
                print(
                    "Custom non-local dispersive material defined by function ",
                    str(self.NL_function),
                )
        # Else, mat should be a list with a function as a first element
        elif isinstance(mat, list) and len(mat) > 0 and callable(mat[0]):
            self.type = "NonLocalModel"
            self.name = "NonLocalModel : " + str(mat[0])
            self.NL_function = mat[0]
            self.params = [mat[i + 1] for i in range(len(mat) - 1)]
            if verbose:
                print(
                    "Custom non-local dispersive material defined by function ",
                    str(self.NL_function),
                )
        # Else, then it's not right
        else:
            print(
                "Please provide a function or a list starting with a function for the model"
            )

    def get_permittivity(self, wavelength):
        _, chi_b, chi_f, _ = self.get_values_nl(wavelength)
        return 1 + chi_b + chi_f

    def get_values_nl(self, wavelength=500):
        # Retrieving the non local material parameters

        w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wavelength

        res = self.NL_function(wavelength, *self.params)
        beta2 = res[0]
        chi_b = res[1]
        chi_f = res[2]
        omega_p = res[3]

        return beta2, chi_b, chi_f, omega_p


def cascade_nl(T, U):
    """
    Cascading Scattering non local matrices
    """
    n = min(np.shape(T)[0], np.shape(U)[0]) - 1
    m = np.shape(T)[0] - n
    p = np.shape(U)[0] - n

    A = T[0:m, 0:m]
    B = T[0:m, m : m + n]
    C = T[m : m + n, 0:m]
    D = T[m : m + n, m : m + n]

    E = U[0:n, 0:n]
    F = U[0:n, n : n + p]
    G = U[n : n + p, 0:n]
    H = U[n : n + p, n : n + p]

    J = np.linalg.inv(np.eye(n, n) - E @ D)
    K = np.linalg.inv(np.eye(n, n) - D @ E)
    matrix = np.vstack(
        (
            np.hstack((A + B @ J @ E @ C, B @ J @ F)),
            np.hstack((G @ K @ C, H + G @ K @ D @ F)),
        )
    )
    return matrix


def NLcoefficient(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure, and takes account the possibility to have NonLocal materials

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    Also, you should not use two adjacent nonlocal material layers, it doesn't work for the moment..
    """

    # In order to get a phase that corresponds to the expected reflected coefficient, we make the height of the upper (lossless) medium vanish. It changes only the phase of the reflection coefficient.
    # The medium may be dispersive. The permittivity and permability of each layer has to be computed each time.
    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)

    Epsilon_mat, Mu_mat = struct.polarizability(wavelength)
    Type = struct.layer_type
    Epsilon = [Epsilon_mat[i] for i in Type]
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0

    if len(struct.thickness) != len(struct.layer_type):
        print(
            f"ArgumentMatchError : layer_type has {len(struct.layer_type)} arguments and thickness has {len(struct.thickness)} arguments"
        )
        return None

    # The boundary conditions will change when the polarization changes. (A demander à Antoine pourquoi)
    if polarization == 0:
        print("Non local materials should be used with polarization = 1 (TM)")
        return 0
    else:
        f = Epsilon

    k_0 = 2 * np.pi / wavelength
    g = len(Type)
    omega_p = [0] * (g)
    chi_b = [0] * (g)
    chi_f = [0] * (g)
    beta2 = [0] * (g)
    for k in range(g):
        if struct.materials[Type[k]].specialType == "NonLocal":
            beta2[k], chi_b[k], chi_f[k], omega_p[k] = struct.materials[
                Type[k]
            ].get_values_nl(wavelength)

    alpha = np.sqrt(Epsilon[0]) * k_0 * np.sin(incidence)
    gamma = np.array(
        np.sqrt([(1 + 0j) * Epsilon[i] * k_0 ** 2 - alpha ** 2 for i in range(g)]),
        dtype=complex,
    )

    T = []
    thickness[0] = 0
    T.append(np.array([[0, 1], [1, 0]], dtype=complex))
    # print(f"Matrice {0} de couche locale (initialisation) \nt vaut : {1., 1.0j}, \n {T[0]}")

    for k in range(g - 1):
        # Stability of square root in complex world :)
        if np.imag(gamma[k + 1]) < 0:
            gamma[k + 1] *= -1

        b1 = gamma[k] / f[k]
        b2 = gamma[k + 1] / f[k + 1]
        # print(f"b1 vaut {b1} \nb2 vaut {b2}")

        # local layer matrix
        if beta2[k] == 0:
            t = np.exp(1j * gamma[k] * thickness[k])
            T.append(np.array([[0, t], [t, 0]], dtype=complex))

            if beta2[k + 1] == 0:
                # local local interface
                T.append(
                    np.array(
                        [[b1 - b2, 2 * b2], [2 * b1, b2 - b1]] / (b1 + b2),
                        dtype=complex,
                    )
                )

            else:
                # local non-local interface
                Kl = np.sqrt(
                    alpha ** 2
                    + (omega_p[k + 1] ** 2 / beta2[k + 1])
                    * (1 / chi_f[k + 1] + 1 / (1 + chi_b[k + 1]))
                )
                omega = (alpha ** 2 / Kl) * (
                    1 / Epsilon[k + 1] - 1 / (1 + chi_b[k + 1])
                )

                T.append(
                    np.array(
                        [
                            [b1 - b2 + 1j * omega, 2 * b2, 2],
                            [2 * b1, b2 - b1 + 1j * omega, 2],
                            [
                                2 * 1j * omega * b1,
                                2 * 1j * omega * b2,
                                b1 + b2 + 1j * omega,
                            ],
                        ]
                        / (b1 + b2 - 1j * omega),
                        dtype=complex,
                    )
                )

        else:  # if beta[k] != 0 :
            Kl = np.sqrt(
                alpha ** 2
                + (omega_p[k] ** 2 / beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]))
            )
            omega = (alpha ** 2 / Kl) * (1 / Epsilon[k] - 1 / (1 + chi_b[k]))
            t = np.exp(1j * gamma[k] * thickness[k])
            l = np.exp(-Kl * thickness[k])
            T.append(
                np.array(
                    [[0, 0, t, 0], [0, 0, 0, l], [t, 0, 0, 0], [0, l, 0, 0]],
                    dtype=complex,
                )
            )

            if beta2[k + 1] == 0:
                # non-local local interface
                T.append(
                    np.array(
                        [
                            [b1 - b2 + 1j * omega, -2, 2 * b2],
                            [
                                -2 * 1j * omega * b1,
                                b1 + b2 + 1j * omega,
                                -2 * 1j * omega * b2,
                            ],
                            [2 * b1, -2, b2 - b1 + 1j * omega],
                        ]
                        / (b1 + b2 - 1j * omega),
                        dtype=complex,
                    )
                )

            else:
                # non-local non-local interface
                print("We can't use cascadage for non local - non local layers (yet)")

    # Last layer
    t = np.exp(1j * gamma[g - 1] * thickness[g - 1])
    T.append(np.array([[0, t], [t, 0]], dtype=complex))
    # print(f"Matrice {2 * g - 1} de couche locale, t vaut : {t} \n {T[2 * g - 1]}")
    # print(f"kl vaut {Kl} \nomega vaut {omega}")

    # INITIALISATION
    A = T[0]  # np.array([[0, 1], [1, 0]], dtype = complex)

    # Cascading scattering matrices
    for p in range(len(T) - 1):  # len(T) - 1 = 2 * g - 1
        A = cascade_nl(A, T[p])
    # Reflection coefficient
    r = A[0, 0]
    # Transmission coefficient
    t = A[1, 0]
    # Energy reflexion coefficient
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient
    T = np.real(abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return r, t, R, T


def NLdispersion(alpha, struct, wavelength, polarization):
    """
    It would probably be better to compute the dispersion relation of a
    multilayered structure, like the determinant of the inverse of the
    scattering matrix. However, strangely enough, for a single interface, it
    just does not work. Even though the coefficients of the scattering matrix
    diverge the determinant does not, so that it does not work to find the
    surface plasmon mode, force instance.

    The present function actually computes the inverse of the modulus of the
    reflection coefficient. Since a mode is a pole of the coefficient, here it
    should be a zero of the resulting function. The determination of the square
    root is modified, so that the modes are not hidden by any cut due to the
    square root.

    Args:
        alpha (complex) : wavevector
        struct (NLStructure) : the object describing the multilayer, including non local materials
        wavelength : the wavelength in vacuum in nanometer
        polarization : 0 for TE, 1 for TM.

    Returns:
        1/abs(r) : inverse of the modulus of the reflection coefficient.

    """

    # In order to get a phase that corresponds to the expected reflected coefficient, we make the height of the upper (lossless) medium vanish. It changes only the phase of the reflection coefficient.
    # The medium may be dispersive. The permittivity and permability of each layer has to be computed each time.
    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)

    Epsilon_mat, Mu_mat = struct.polarizability(wavelength)
    Type = struct.layer_type
    Epsilon = [Epsilon_mat[i] for i in Type]
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0

    if len(struct.thickness) != len(struct.layer_type):
        print(
            f"ArgumentMatchError : layer_type has {len(struct.layer_type)} arguments and thickness has {len(struct.thickness)} arguments"
        )
        return None

    # The boundary conditions will change when the polarization changes. (A demander à Antoine pourquoi)
    if polarization == 0:
        print("Non local materials should be used with polarization = 1 (TM)")
        return 0
    else:
        f = Epsilon

    k0 = 2 * np.pi / wavelength
    g = len(Type)
    omega_p = [0] * (g)
    chi_b = [0] * (g)
    chi_f = [0] * (g)
    beta2 = [0] * (g)
    for k in range(g):
        if struct.materials[Type[k]].specialType == "NonLocal":
            beta2[k], chi_b[k], chi_f[k], omega_p[k] = struct.materials[
                Type[k]
            ].get_values_nl(wavelength)

    gamma = np.array(
        np.sqrt([(1 + 0j) * Epsilon[i] * k0 ** 2 - alpha ** 2 for i in range(g)]),
        dtype=complex,
    )

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))

    # Changing the determination of the square root in the external medium
    # to better see the structure of the complex plane.
    gamma[0] = gamma[0] * (1 - 2 * (np.angle(gamma[0]) < -np.pi / 5))
    gamma[g - 1] = gamma[g - 1] * (1 - 2 * (np.angle(gamma[g - 1]) < -np.pi / 5))

    # First S Matrix
    T = []
    thickness[0] = 0
    T.append(np.array([[0, 1], [1, 0]], dtype=complex))

    for k in range(g - 1):

        b1 = gamma[k] / f[k]
        b2 = gamma[k + 1] / f[k + 1]
        # print(f"b1 vaut {b1} \nb2 vaut {b2}")

        # local layer matrix
        if beta2[k] == 0:
            t = np.exp(1j * gamma[k] * thickness[k])
            T.append(np.array([[0, t], [t, 0]], dtype=complex))

            if beta2[k + 1] == 0:
                # local local interface
                T.append(
                    np.array(
                        [[b1 - b2, 2 * b2], [2 * b1, b2 - b1]] / (b1 + b2),
                        dtype=complex,
                    )
                )

            else:
                # local non-local interface
                Kl = np.sqrt(
                    alpha ** 2
                    + (omega_p[k + 1] ** 2 / beta2[k + 1])
                    * (1 / chi_f[k + 1] + 1 / (1 + chi_b[k + 1]))
                )
                omega = (alpha ** 2 / Kl) * (
                    1 / Epsilon[k + 1] - 1 / (1 + chi_b[k + 1])
                )

                T.append(
                    np.array(
                        [
                            [b1 - b2 + 1j * omega, 2 * b2, 2],
                            [2 * b1, b2 - b1 + 1j * omega, 2],
                            [
                                2 * 1j * omega * b1,
                                2 * 1j * omega * b2,
                                b1 + b2 + 1j * omega,
                            ],
                        ]
                        / (b1 + b2 - 1j * omega),
                        dtype=complex,
                    )
                )
                # print(f"Matrice {2 * k + 1} de couche locale \nt vaut : {t} \n {T[2 * k + 1]}")

        else:  # if beta[k] != 0 :
            Kl = np.sqrt(
                alpha ** 2
                + (omega_p[k] ** 2 / beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]))
            )
            omega = (alpha ** 2 / Kl) * (1 / Epsilon[k] - 1 / (1 + chi_b[k]))
            t = np.exp(1j * gamma[k] * thickness[k])
            l = np.exp(-Kl * thickness[k])
            T.append(
                np.array(
                    [[0, 0, t, 0], [0, 0, 0, l], [t, 0, 0, 0], [0, l, 0, 0]],
                    dtype=complex,
                )
            )
            # print(f"Matrice {2 * k + 1} de couche non-locale \nt vaut : {t} \nl vaut : {l} \n, {T[2 * k + 1]}")

            if beta2[k + 1] == 0:
                # non-local local interface
                T.append(
                    np.array(
                        [
                            [b1 - b2 + 1j * omega, -2, 2 * b2],
                            [
                                -2 * 1j * omega * b1,
                                b1 + b2 + 1j * omega,
                                -2 * 1j * omega * b2,
                            ],
                            [2 * b1, -2, b2 - b1 + 1j * omega],
                        ]
                        / (b1 + b2 - 1j * omega),
                        dtype=complex,
                    )
                )

            else:
                # non-local non-local interface
                print("We can't use cascadage for non local - non local layers (yet)")

    # Last layer
    t = np.exp(1j * gamma[g - 1] * thickness[g - 1])
    T.append(np.array([[0, t], [t, 0]], dtype=complex))

    # INITIALISATION
    A = T[0]

    # Cascading scattering matrices
    for p in range(len(T) - 1):
        A = cascade_nl(A, T[p])
    # Reflection coefficient
    r = A[0, 0]

    return 1 / np.abs(r)


def NLsteepest(start, tol, step_max, struct, wl, pol):
    """NL Steepest descent to find a zero of the `dispersion`
    function. The advantage of looking for a zero is that you
    know when the algorithm can stop (when the value of the function
    is smaller than `tol`).

    Args:
        start (complex): effective index where the descent starts
        tol (real): when dispersion is smaller than tol, the
                    descent stops.
        step_max (integer): maximum number of steps allowed
        struct (Structure): the object describing the multilayer
        wl (float): wavelength in vacuum
        pol: 0 for TE, 1 for TM

    Returns:

        (float) : last effective index reached at the end of the descent

    """

    k_0 = 2 * np.pi / wl
    z = start * k_0
    delta = abs(z) * 0.001
    dz = 0.01 * delta
    step = 0
    current = NLdispersion(z, struct, wl, pol)

    while (current > tol) and (step < step_max):

        grad = (
            NLdispersion(z + dz, struct, wl, pol)
            - current
            #        -dispersion(z-dz,struct,wl,pol)
            + 1j
            * (
                NLdispersion(z + 1j * dz, struct, wl, pol)
                #        -dispersion(z-1j*dz,struct,wl,pol))
                - current
            )
        ) / (dz)

        if abs(grad) != 0:
            z_new = z - delta * grad / abs(grad)
        else:
            # We have a finishing condition not linked to the gradient
            # So if we meet a gradient of 0, we just divide the step by two
            delta = delta / 2.0
            z_new = z

        value_new = NLdispersion(z_new, struct, wl, pol)
        if value_new > current:
            # The path not taken
            delta = delta / 2.0
            dz = dz / 2.0
        else:
            current = value_new
            z = z_new
        #        print("Step", step, z,current)
        step = step + 1

    # print("End of the loop")
    if step == step_max:
        print(
            "Warning: maximum number of steps reached. Final n_eff:",
            z / k_0,
            "final value",
            current,
        )

    return z / k_0


def NLguided_modes(
    struct, wavelength, polarization, neff_min, neff_max, initial_points=40
):
    """This function explores the complex plane, looking for zeros of the
    dispersion relation. It does so by launching a NLsteepest descent for a number
    `initial_points` of points on the real axis between neff_min and neff_max.


    Args:
        struct (Structure): object describing the multilayer
        wavelength (float): wavelength in nm
        polarization: 0 for TE, 1 for TM
        neff_min: minimum value of the effective index expected
        neff_max: maximum value of the effective index expected

    Returns:
        modes (list, complex): complex effective index identified as
                            solutions of the dispersion relation.

    """

    tolerance = 1e-10
    #    initial_points = 40
    k_0 = 2 * np.pi / wavelength
    neff_start = np.linspace(neff_min, neff_max, initial_points, dtype=complex)
    modes = []
    for neff in neff_start:
        solution = NLsteepest(neff, tolerance, 1000, struct, wavelength, polarization)

        if len(modes) == 0:
            modes.append(solution)
        elif min(abs(modes - solution)) > 1e-5 * k_0:
            modes.append(solution)

    return modes


def NLcomplex_map(
    struct, wavelength, polarization, real_bounds, imag_bounds, n_real, n_imag
):
    """Maps the function `dispersion` supposed to vanish when the dispersion
    relation is satisfied.

    Args:
        struct (Structure): object Structure describing the multilayer
        wavelength: wavelength in vacuum (in nm)
        polarization: 0 for TE, 1 for TM
        real_bounds: a list giving the bounds of the effective index
                     real part [n_min,n_max], defining the zone to
                     explore.
        imag_bounds: a list giving the bounds of the effective index
                     imaginary part.
        n_real: number of points horizontally (real part)
        n_imag: number of points vertically (imaginary part)

    Returns:
        X (1D numpy array): values of the real part of the effective index
        Y (1D numpy array): values of the imaginary part of the effective index
        T (2D numpy array): values of the dispersion function

    In order to visualize the map, just use :
        import matplotlib.pyplot as plt
        plt.contourf(X,Y,np.sqrt(np.real(T)))
        plt.show()
    """

    k_0 = 2 * np.pi / wavelength
    X = np.linspace(real_bounds[0], real_bounds[1], n_real)
    Y = np.linspace(imag_bounds[0], imag_bounds[1], n_imag)
    xa, xb = np.meshgrid(X * k_0, Y * k_0)
    M = xa + 1j * xb

    T = np.zeros((n_real, n_imag), dtype=complex)
    for k in range(n_real):
        for l in range(n_imag):
            T[k, l] = 1 / NLdispersion(M[k, l], struct, wavelength, polarization)

    return X, Y, T


def NLfollow_guided_modes(
    struct,
    wavelength_list,
    polarization,
    neff_min,
    neff_max,
    format="n",
    initial_points=40,
    plot=True,
):
    """This function explores the complex plane, looking for zeros of the
    dispersion relation. It does so by launching a NLsteepest descent for a number
    `initial_points` of points on the real axis between neff_min and neff_max.


    Args:
        struct (Structure): object describing the multilayer, including non local materials
        wavelength_list (float): wavelengths in nm
        polarization: 0 for TE, 1 for TM
        neff_min: minimum value of the effective index expected
        neff_max: maximum value of the effective index expected
        format: index output format, n for effective index, wav for wavelength, k for wavevector

    Returns:
        modes (list of list, complex): complex effective indices identified as
                            solutions of the dispersion relation for all wavelengths.

    """

    if not format in ["n", "k", "wav"]:
        print("Unknown index format: accepted values or n, k and wav")

    tolerance = 1e-10
    #    initial_points = 40
    wavelength = wavelength_list[0]
    k_0 = 2 * np.pi / wavelength
    neff_start = np.linspace(neff_min, neff_max, initial_points, dtype=complex)

    # Finding the first modes, that we will then follow
    first_modes = []
    for neff in neff_start:
        solution = NLsteepest(neff, tolerance, 1000, struct, wavelength, polarization)
        #        print(solution)
        if len(first_modes) == 0:
            first_modes.append(solution)
        elif min(abs(first_modes - solution)) > 1e-5 * k_0:
            first_modes.append(solution)
    modes = [first_modes]

    # Following these modes for all the wavelength range
    for i in range(1, len(wavelength_list)):
        wavelength = wavelength_list[i]
        k_0 = 2 * np.pi / wavelength
        neff_start = np.array(modes[-1]).flatten()
        new_modes = []
        if len(neff_start) > 1:
            for neff in neff_start:
                solution = NLsteepest(
                    neff, tolerance, 1000, struct, wavelength, polarization
                )
                if len(new_modes) == 0:
                    new_modes.append(solution)
                elif min(abs(new_modes - solution)) > 1e-5 * k_0:
                    new_modes.append(solution)
        else:
            neff = neff_start
            solution = NLsteepest(
                neff, tolerance, 1000, struct, wavelength, polarization
            )
            if len(new_modes) == 0:
                new_modes.append(solution)
            elif min(abs(new_modes - solution)) > 1e-5 * k_0:
                new_modes.append(solution)

        modes.append(np.array(new_modes).flatten())

    if format == "k":
        for i in range(len(modes)):
            modes[i] = np.array(modes[i]) * 2 * np.pi / wavelength_list[i]
    elif format == "wav":
        for i in range(len(modes)):
            modes[i] = wavelength_list[i] / np.array(modes[i])

    mode_filled = np.array(list(itertools.zip_longest(*modes, fillvalue=0)))
    mode_filled = mode_filled.T

    # The following bit links modes together, for ease of use
    # However, it only follows the modes that exist at the smallest wavelength
    follow_modes = []
    nb_mode = np.shape(mode_filled)[1]
    for shift in range(nb_mode // 2):
        index = (
            nb_mode // 2 + shift
        )  # starting with the middle mode because it's the usually the most stable one
        mode = [mode_filled[0, index]]
        i = 1
        while i < len(mode_filled) and index >= 0 and index < nb_mode:
            res = np.abs(mode[-1] - mode_filled[i, index - 1 : index + 2])
            if len(res) == 3:
                a, b, c = res
                if a < b and a < c:
                    index = index - 1
                elif c < a and c < b:
                    index = index + 1
                mode.append(mode_filled[i, index])
            else:
                break
            i += 1
        follow_modes.append(mode)

        if shift > 0:
            index = nb_mode // 2 - shift
            mode = [mode_filled[0, index]]
            i = 1
            while i < len(mode_filled) and index >= 0 and index < nb_mode:
                res = np.abs(mode[-1] - mode_filled[i, index - 1 : index + 2])
                if len(res) == 3:
                    a, b, c = res
                    if a < b and a < c:
                        index = index - 1
                    elif c < a and c < b:
                        index = index + 1
                    mode.append(mode_filled[i, index])
                else:
                    break
                i += 1
            follow_modes.append(mode)

    follow_modes = np.array(list(itertools.zip_longest(*follow_modes, fillvalue=0)))
    # Because we are following modes, we add 0 when the modes disappear

    if plot:
        modes_no_zero = []
        for i in range(np.shape(follow_modes)[1]):
            j = 0
            while j < np.shape(follow_modes)[0] and follow_modes[j, i] != 0:
                j += 1
            modes_no_zero.append(follow_modes[: j - 1, i])

        for i in range(len(modes_no_zero)):
            if format == "n":
                plt.plot(
                    wavelength_list[: len(modes_no_zero[i])],
                    wavelength_list[: len(modes_no_zero[i])] / modes_no_zero[i],
                )
                plt.xlabel("Wavelength in vacuum (nm)")
                plt.ylabel("Effective Wavelength (nm)")
            elif format == "wav":
                plt.plot(wavelength_list[: len(modes_no_zero[i])], modes_no_zero[i])
                plt.xlabel("Wavelength in vacuum (nm)")
                plt.ylabel("Effective Wavelength (nm)")
            elif format == "k":
                plt.plot(
                    2 * np.pi / wavelength_list[: len(modes_no_zero[i])],
                    modes_no_zero[i],
                )
                plt.xlabel("Wavevector in vacuum (nm-1)")
                plt.ylabel("Effective Wavevector (nm-1)")
        plt.show()

    return modes, follow_modes
