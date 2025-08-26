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

# TODO: add absorption


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
        custom function must return: beta2 (m s-1), chi_b (non dim), chi_f (non dim), omega_p (Hz, or a different order of magnitude, to work with nm later, see NLcoefficient KL)
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
        np.sqrt([(1 + 0j) * Epsilon[i] * k_0**2 - alpha**2 for i in range(g)]),
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
                    alpha**2
                    + (omega_p[k + 1] ** 2 / beta2[k + 1])
                    * (1 / chi_f[k + 1] + 1 / (1 + chi_b[k + 1]))
                )
                omega = (alpha**2 / Kl) * (1 / Epsilon[k + 1] - 1 / (1 + chi_b[k + 1]))

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
                alpha**2
                + (omega_p[k] ** 2 / beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]))
            )
            omega = (alpha**2 / Kl) * (1 / Epsilon[k] - 1 / (1 + chi_b[k]))
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


def intermediaire(T, U):
    """
    Cascading operation for non local materials
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
            np.hstack((K @ C, K @ D @ F)),
            np.hstack((J @ E @ C, J @ F)),
        )
    )
    return matrix


def fields_NL_TL(struct, beam, window):
    """
    Computes the electric (TE polarization) or magnetic (TM) field inside
    a multilayered structure with possibly NonLocal materials illuminated by a
    gaussian beam, and uses them to compute the rest of the fields (Hx,Hz in TE polarization
    /Ex,Ez in TM).
    To be precise, the derived fields need to be divided by omega*µ0 in TE
    and omega*epsilon_0 in TM. There a missing - sign in TM, too.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        beam (Beam): description of the incidence beam
        window (Window): description of the simulation domain

    Returns:
        Hyn_t (np.array): a matrix with the complex amplitude of the field
        Hyn_l (np.array): a matrix with the complex amplitude of Hy longitudinal
        Exn_t (np.array): a matrix with the complex amplitude of Ex transverse
        Exn_l (np.array): a matrix with the complex amplitude of Ex longitudinal
        Ezn_l (np.array): a matrix with the complex amplitude of Ez longitudinal
        rho_n (np.array): a matrix with the complex amplitude of charge density
        jfx_n (np.array): a matrix with the complex amplitude of current density along x
        jfz_n (np.array): a matrix with the complex amplitude of current density along z

    Also, you should not use two adjacent nonlocal material layers, it doesn't work for the moment..
    """

    # Wavelength in vacuum.
    lam = beam.wavelength
    # Computation of all the permittivities/permeabilities
    Epsilon, Mu = struct.polarizability(lam)
    thickness = np.array(struct.thickness)
    w = beam.waist
    pol = beam.polarization
    d = window.width
    theta = beam.incidence
    C = window.C
    ny = np.floor(thickness / window.py)
    nx = window.nx
    Type = struct.layer_type
    print("Pixels vertically:", int(sum(ny)))

    # Number of modes retained for the description of the field
    # so that the last mode has an amplitude < 1e-3 - you may want
    # to change it if the structure present reflexion coefficients
    # that are subject to very swift changes with the angle of incidence.

    nmod = int(np.floor(0.83660 * d / w))

    # Normalization with respect to the window size
    l = lam / d
    w = w / d
    thickness = thickness / d

    if pol == 0:
        print("Non local materials should be used with polarization = 1 (TM)")
        return 0
    else:
        f = Epsilon
    # Wavevector in vacuum, no dimension
    k_0 = 2 * np.pi / l
    # Initialization of the fields component
    Hyn_t = np.zeros((int(sum(ny)), int(nx)))
    Hyn_l = np.zeros((int(sum(ny)), int(nx)))
    Exn_t = np.zeros((int(sum(ny)), int(nx)))
    Exn_l = np.zeros((int(sum(ny)), int(nx)))
    Ezn_t = np.zeros((int(sum(ny)), int(nx)))
    Ezn_l = np.zeros((int(sum(ny)), int(nx)))
    rhon = np.zeros((int(sum(ny)), int(nx)))
    jfx_n = np.zeros((int(sum(ny)), int(nx)))
    jfz_n = np.zeros((int(sum(ny)), int(nx)))
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1

    # non local parameters
    omega_p = [0] * (g + 1)
    chi_b = [0] * (g + 1)
    chi_f = [0] * (g + 1)
    beta2 = [0] * (g + 1)

    # helping parameters
    omega_p2 = [0] * (g + 1)  # wp^2
    omega2_beta2 = [0] * (g + 1)  # wp^2 / beta^2
    k_nl2 = [0] * (g + 1)

    for k in range(g):
        if struct.materials[Type[k]].specialType == "NonLocal":
            beta2[k], chi_b[k], chi_f[k], omega_p[k] = struct.materials[
                Type[k]
            ].get_values_nl(lam)
            omega_p[k] = omega_p[k] * d  # omega_p is normalized too
            # Compute helping parameters
            omega_p2[k] = omega_p[k] ** 2
            omega2_beta2[k] = omega_p2[k] / beta2[k]
            k_nl2[k] = (omega2_beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]))

    # Amplitude of the different modes
    nmodvect = np.arange(-nmod, nmod + 1)
    # First factor makes the gaussian beam, the second one the shift
    # a constant phase is missing, it's just a change in the time origin.
    X = np.exp(-(w**2) * np.pi**2 * nmodvect**2) * np.exp(
        -2 * 1j * np.pi * nmodvect * C
    )

    # Scattering matrix corresponding to no interface.
    T = [0] * (2 * g + 2)
    T[0] = np.array([[0, 1], [1, 0]], dtype=complex)

    layer_k = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2)

    for nm in np.arange(2 * nmod + 1):

        alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k_0 * np.sin(
            theta
        ) + 2 * np.pi * (nm - nmod)

        gamma = [np.sqrt(layer_k[i] ** 2 - alpha**2) for i in range(g + 1)]

        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        # Choosing the correct determination of the square root
        # (positive imaginary part)
        if g > 2:
            im_sign = np.imag(gamma[1 : g - 1]) < 0
            gamma[1 : g - 1] = gamma[1 : g - 1] * (1 - 2 * im_sign)
        if (
            np.real(Epsilon[Type[g]]) < 0
            and np.real(Mu[Type[g]]) < 0
            and np.real(np.sqrt(layer_k[g] ** 2 - alpha**2)) != 0
        ):
            gamma[g] = -gamma[g]
        gf = gamma / f[Type]

        for k in range(g):

            b1 = gf[k]
            b2 = gf[k + 1]
            if beta2[k] == 0:  # local layer
                t = np.exp(1j * gamma[k] * thickness[k])
                T[2 * k + 1] = np.array([[0, t], [t, 0]], dtype=complex)

                if beta2[k + 1] == 0:  # local-local interface
                    T[2 * k + 2] = np.array(
                        [[b1 - b2, 2 * b2], [2 * b1, b2 - b1]], dtype=complex
                    ) / (b1 + b2)

                else:  # local-non local interface
                    Kl = np.sqrt(alpha**2 + k_nl2[k + 1])
                    omega = (alpha**2 / Kl) * (
                        1 / Epsilon[Type[k + 1]] - 1 / (1 + chi_b[k + 1])
                    )
                    T[2 * k + 2] = np.array(
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

            else:  # non-local layer
                Kl = np.sqrt(alpha**2 + k_nl2[k])
                omega = (alpha**2 / Kl) * (1 / Epsilon[Type[k]] - 1 / (1 + chi_b[k]))
                t = np.exp(1j * gamma[k] * thickness[k])
                l = np.exp(-1 * Kl * thickness[k])
                T[2 * k + 1] = np.array(
                    [[0, 0, t, 0], [0, 0, 0, l], [t, 0, 0, 0], [0, l, 0, 0]],
                    dtype=complex,
                )

                if beta2[k + 1] == 0:  # non-local - local interface
                    T[2 * k + 2] = np.array(
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

                else:
                    # non-local non-local interface
                    print(
                        "We can't use cascadage for non local - non local layers (yet)"
                    )
                    return None

        # last layer
        t = np.exp(1j * gamma[g] * thickness[g])
        T[2 * g + 1] = np.array([[0, t], [t, 0]], dtype=complex)

        # Once the scattering matrixes have been prepared, now let us combine them

        H = [0] * (2 * g + 1)
        A = [0] * (2 * g + 1)

        H[0] = T[2 * g + 1]
        A[0] = T[0]

        for k in range(len(T) - 2):
            A[k + 1] = cascade_nl(A[k], T[k + 1])
            H[k + 1] = cascade_nl(T[len(T) - k - 2], H[k])

        # And let us compute the intermediate coefficients from the scattering matrixes

        I = [0] * (len(T))
        for k in range(len(T) - 1):
            I[k] = intermediaire(A[k], H[len(T) - k - 2])

        I[2 * g + 1] = np.zeros((2, 2))

        h = 0
        t = 0

        # Computation of the fields in the different layers for one mode (plane wave)
        Hy_t = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        Ex_t = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        Ex_l = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        Ez_t = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        Ez_l = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        rho = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        jfx = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        jfz = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        for k in range(g + 1):
            for m in range(int(ny[k])):
                h = h + float(thickness[k]) / ny[k]
                # The expression for the field used here is based on the assumption
                # that the structure is illuminated from above only, with an Amplitude
                # of 1 for the incident wave. If you want only the reflected
                # field, take off the second term.

                if beta2[k] == 0:  # local layer

                    H1 = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h)
                    H2 = I[2 * k + 1][1, 0] * np.exp(1j * gamma[k] * (thickness[k] - h))

                    Hy_t[t, 0] = H1 + H2

                    Ex_t[t, 0] = -1 * gf[k] * (H1 - H2)
                    Ez_t[t, 0] = alpha * Hy_t[t, 0] / f[Type[k]]

                    # No longitudinal fields
                    Ex_l[t, 0] = 0
                    Ez_l[t, 0] = 0

                    # div(E) = 0, no charge density
                    rho[t, 0] = 0

                    jfx[t, 0] = 0
                    jfz[t, 0] = 0

                    t += 1

                elif beta2[k] != 0:  # non-local layer

                    Kl = np.sqrt(alpha**2 + k_nl2[k])

                    H1 = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h)
                    H2 = I[2 * k + 1][2, 0] * np.exp(1j * gamma[k] * (thickness[k] - h))
                    H3 = I[2 * k][1, 0] * np.exp(-1 * Kl * h)
                    H4 = I[2 * k + 1][3, 0] * np.exp(-1 * Kl * (thickness[k] - h))

                    Hy_t[t, 0] = H1 + H2

                    Ex_t[t, 0] = -1 * gf[k] * (H1 - H2)
                    Ez_t[t, 0] = alpha * Hy_t[t, 0] / f[Type[k]]

                    Ex_l[t, 0] = H3 + H4
                    Ez_l[t, 0] = 1j * Kl / alpha * (H3 - H4)

                    rho[t, 0] = (
                        (-1 * Kl**2 / alpha + 1j * alpha) * (1 + chi_b[k]) * Ex_l[t, 0]
                    )

                    prefac = (
                        (1 + chi_b[k])
                        * beta2[k]
                        / omega_p2[k]
                        * (alpha**2 + 1j * Kl**2)
                    )
                    Ex_tot = Ex_t[t, 0] + Ex_l[t, 0]
                    jfx[t, 0] = -1j * chi_f[k] * (Ex_tot + (prefac * Ex_l[t, 0]))
                    Ez_tot = Ez_t[t, 0] + Ez_l[t, 0]
                    jfz[t, 0] = -1j * chi_f[k] * (Ez_tot + (prefac * Ez_l[t, 0]))
                    t += 1
            h = 0

        Hy_t = Hy_t * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Ex_t = Ex_t * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Ex_l = Ex_l * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Ez_t = Ez_t * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Ez_l = Ez_l * np.exp(1j * alpha * np.arange(0, nx) / nx)
        rho = rho * np.exp(1j * alpha * np.arange(0, nx) / nx)
        jfx = jfx * np.exp(1j * alpha * np.arange(0, nx) / nx)
        jfz = jfz * np.exp(1j * alpha * np.arange(0, nx) / nx)

        Hyn_t = Hyn_t + X[int(nm)] * Hy_t
        Exn_t = Exn_t + X[int(nm)] * Ex_t
        Exn_l = Exn_l + X[int(nm)] * Ex_l
        Ezn_t = Ezn_t + X[int(nm)] * Ez_t
        Ezn_l = Ezn_l + X[int(nm)] * Ez_l
        rhon = rho + X[int(nm)] * rho
        jfx_n = jfx_n + X[int(nm)] * jfx
        jfz_n = jfz_n + X[int(nm)] * jfz

    return Hyn_t, Hyn_l, Exn_t, Exn_l, Ezn_t, Ezn_l, rhon, jfx_n, jfz_n


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

    gamma = np.array(
        np.sqrt([(1 + 0j) * Epsilon[i] * k_0**2 - alpha**2 for i in range(g)]),
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
                    alpha**2
                    + (omega_p[k + 1] ** 2 / beta2[k + 1])
                    * (1 / chi_f[k + 1] + 1 / (1 + chi_b[k + 1]))
                )
                omega = (alpha**2 / Kl) * (1 / Epsilon[k + 1] - 1 / (1 + chi_b[k + 1]))

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
                alpha**2
                + (omega_p[k] ** 2 / beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]))
            )
            omega = (alpha**2 / Kl) * (1 / Epsilon[k] - 1 / (1 + chi_b[k]))
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
