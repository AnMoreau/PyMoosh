"""
This file contains all functions that helps with computing gradients
"""

import numpy as np
import copy
from PyMoosh.classes import conv_to_nm, Structure
from PyMoosh.alt_methods import coefficient_A, coefficient_T
from PyMoosh.core import coefficient_S


def coefficient_with_grad_A(
    struct,
    wavelength,
    incidence,
    polarization,
    mode="value",
    i_change=-1,
    saved_mat=None,
):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the (true) Abeles matrix formalism.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM
        mode (string): "value" to compute r, t, R, T, "grad" to compute them with a small parameter shift
        i_change (int): the layer where there is a modification
        saved_mat (arrays): the matrices used to compute the coefficients


    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    """

    if mode == "grad" and saved_mat is None:
        print(
            "Not giving the abeles matrices to compute the gradient leads to regular computation"
        )
        return coefficient_A(struct, wavelength, incidence, polarization)

    # In order to get a phase that corresponds to the expected reflected coefficient,
    # we make the height of the upper (lossless) medium vanish. It changes only the
    # phase of the reflection coefficient.

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)
    Epsilon, Mu = struct.polarizability(wavelength)
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g) * alpha**2)
    # Be cautious if the upper medium is a negative index one.
    if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
        gamma[0] = -gamma[0]

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))
    # Outgoing wave condition for the last medium
    if (
        np.real(Epsilon[Type[g - 1]]) < 0
        and np.real(Mu[Type[g - 1]]) < 0
        and np.real(np.sqrt(Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2))
        != 0
    ):
        gamma[g - 1] = -np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2
        )
    else:
        gamma[g - 1] = np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2
        )

    if mode == "value":
        T = np.zeros(((g - 1, 2, 2)), dtype=complex)
        for k in range(g - 1):
            # Layer scattering matrix
            c = np.cos(gamma[k] * thickness[k])
            s = np.sin(gamma[k] * thickness[k])

            T[k] = [[c, -f[Type[k]] / gamma[k] * s], [gamma[k] / f[Type[k]] * s, c]]
        # Once the scattering matrixes have been prepared, now let us combine them

        A = np.empty((len(T) + 1, 2, 2), dtype=complex)
        B = np.empty((len(T) + 1, 2, 2), dtype=complex)
        A[0] = np.array([[1, 0], [0, 1]])
        B[0] = np.array([[1, 0], [0, 1]])
        for i in range(1, T.shape[0] + 1):
            A[i] = T[i - 1] @ A[i - 1]
            B[i] = B[i - 1] @ T[-i]

        a = A[-1][0, 0]
        b = A[-1][0, 1]
        c = A[-1][1, 0]
        d = A[-1][1, 1]

        amb = a - 1.0j * gamma[0] / f[Type[0]] * b
        apb = a + 1.0j * gamma[0] / f[Type[0]] * b
        cmd = c - 1.0j * gamma[0] / f[Type[0]] * d
        cpd = c + 1.0j * gamma[0] / f[Type[0]] * d
        # reflection coefficient of the whole structure

        r = -(cmd + 1.0j * gamma[-1] / f[Type[-1]] * amb) / (
            cpd + 1.0j * gamma[-1] / f[Type[-1]] * apb
        )
        # transmission coefficient of the whole structure
        t = a * (r + 1) + 1.0j * gamma[0] / f[Type[0]] * b * (r - 1)
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(
            abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]])
        )

        return r, t, R, T, A, B

    if mode == "grad" and len(saved_mat) == 2:
        A, B = saved_mat
        T = np.zeros((2, 2), dtype=complex)
        # Layer scattering matrix
        c = np.cos(gamma[i_change] * thickness[i_change])
        s = np.sin(gamma[i_change] * thickness[i_change])

        T = [
            [c, -f[Type[i_change]] / gamma[i_change] * s],
            [gamma[i_change] / f[Type[i_change]] * s, c],
        ]

        res = B[-i_change - 2] @ T @ A[i_change]

        a = res[0, 0]
        b = res[0, 1]
        c = res[1, 0]
        d = res[1, 1]

        amb = a - 1.0j * gamma[0] / f[Type[0]] * b
        apb = a + 1.0j * gamma[0] / f[Type[0]] * b
        cmd = c - 1.0j * gamma[0] / f[Type[0]] * d
        cpd = c + 1.0j * gamma[0] / f[Type[0]] * d
        # reflection coefficient of the whole structure

        r = -(cmd + 1.0j * gamma[-1] / f[Type[-1]] * amb) / (
            cpd + 1.0j * gamma[-1] / f[Type[-1]] * apb
        )
        # transmission coefficient of the whole structure
        t = a * (r + 1) + 1.0j * gamma[0] / f[Type[0]] * b * (r - 1)
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(
            abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]])
        )

        return r, t, R, T

    else:
        print("I do not understand what you are trying to do.")


def coefficient_with_grad_T(
    struct,
    wavelength,
    incidence,
    polarization,
    mode="value",
    i_change=-1,
    saved_mat=None,
):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the Transfer matrix formalism.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM
        mode (string): "value" to compute r, t, R, T, "grad" to compute them with a small parameter shift
        i_change (int): the layer where there is a modification
        saved_mat (arrays): the matrices used to compute the coefficients

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    """

    if mode == "grad" and saved_mat is None:
        print(
            "Not giving the T matrices to compute the gradient leads to regular computation"
        )
        return coefficient_T(struct, wavelength, incidence, polarization)

    # In order to get a phase that corresponds to the expected reflected coefficient,
    # we make the height of the upper (lossless) medium vanish. It changes only the
    # phase of the reflection coefficient.

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)
    Epsilon, Mu = struct.polarizability(wavelength)
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g) * alpha**2)
    # Be cautious if the upper medium is a negative index one.
    if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
        gamma[0] = -gamma[0]

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))
    # Outgoing wave condition for the last medium
    if (
        np.real(Epsilon[Type[g - 1]]) < 0
        and np.real(Mu[Type[g - 1]]) < 0
        and np.real(np.sqrt(Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2))
        != 0
    ):
        gamma[g - 1] = -np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2
        )
    else:
        gamma[g - 1] = np.sqrt(
            Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k0**2 - alpha**2
        )

    if mode == "value":
        T = np.zeros(((2 * g - 2, 2, 2)), dtype=complex)
        for k in range(g - 1):
            sum = gamma[k] / f[Type[k]] + gamma[k + 1] / f[Type[k + 1]]
            dif = gamma[k] / f[Type[k]] - gamma[k + 1] / f[Type[k + 1]]
            # print("pop", gamma[k], gamma[k+1])
            # print(sum, dif)
            # Layer transfer matrix
            T[2 * k] = (
                f[Type[k + 1]]
                / (2 * gamma[k + 1])
                * np.array([[sum, -dif], [-dif, sum]])
            )

            phase = 1.0j * gamma[k + 1] * thickness[k + 1]
            # Layer propagation matrix
            T[2 * k + 1] = [[np.exp(-phase), 0], [0, np.exp(phase)]]
        # Once the scattering matrixes have been prepared, now let us combine them

        A = np.empty((len(T) + 1, 2, 2), dtype=complex)
        B = np.empty((len(T) + 1, 2, 2), dtype=complex)
        A[0] = np.array([[1, 0], [0, 1]])
        B[0] = np.array([[1, 0], [0, 1]])
        for i in range(1, T.shape[0] + 1):
            A[i] = A[i - 1] @ T[i - 1]
            B[i] = T[-i] @ B[i - 1]
        r = -A[-1][1, 0] / A[-1][0, 0]
        # transmission coefficient of the whole structure
        t = A[-1][1, 1] - (A[-1][1, 0] * A[-1][0, 1]) / A[-1][0, 0]
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(
            abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]])
        )

        return r, t, R, T, A, B

    if mode == "grad" and len(saved_mat) == 2:
        A, B = saved_mat

        if 0 < i_change:
            T = np.zeros(((3, 2, 2)), dtype=complex)

            sum = (
                gamma[i_change - 1] / f[Type[i_change - 1]]
                + gamma[i_change] / f[Type[i_change]]
            )
            dif = (
                gamma[i_change - 1] / f[Type[i_change - 1]]
                - gamma[i_change] / f[Type[i_change]]
            )
            # Layer transfer matrix
            T[0] = (
                f[Type[i_change]]
                / (2 * gamma[i_change])
                * np.array([[sum, -dif], [-dif, sum]])
            )

            phase = 1.0j * gamma[i_change] * thickness[i_change]
            # Layer propagation matrix
            T[1] = [[np.exp(-phase), 0], [0, np.exp(phase)]]

            sum = (
                gamma[i_change] / f[Type[i_change]]
                + gamma[i_change + 1] / f[Type[i_change + 1]]
            )
            dif = (
                gamma[i_change] / f[Type[i_change]]
                - gamma[i_change + 1] / f[Type[i_change + 1]]
            )
            # print("pop", gamma[k], gamma[k+1])
            # print(sum, dif)
            # Layer transfer matrix
            T[2] = (
                f[Type[i_change + 1]]
                / (2 * gamma[i_change + 1])
                * np.array([[sum, -dif], [-dif, sum]])
            )
            # Once the scattering matrixes have been prepared, now let us combine them

            res = A[i_change * 2 - 2] @ T[0] @ T[1] @ T[2] @ B[-(i_change) * 2 - 2]

        else:
            T = np.zeros(((2, 2, 2)), dtype=complex)

            phase = 1.0j * gamma[i_change] * thickness[i_change]
            # Layer propagation matrix
            T[0] = [[np.exp(-phase), 0], [0, np.exp(phase)]]

            sum = (
                gamma[i_change] / f[Type[i_change]]
                + gamma[i_change + 1] / f[Type[i_change + 1]]
            )
            dif = (
                gamma[i_change] / f[Type[i_change]]
                - gamma[i_change + 1] / f[Type[i_change + 1]]
            )
            # print("pop", gamma[k], gamma[k+1])
            # print(sum, dif)
            # Layer transfer matrix
            T[1] = (
                f[Type[i_change + 1]]
                / (2 * gamma[i_change + 1])
                * np.array([[sum, -dif], [-dif, sum]])
            )
            # Once the scattering matrixes have been prepared, now let us combine them
            res = A[i_change * 2] @ T[0] @ T[1] @ B[-(i_change) * 2 - 2]

        # print(T)
        # print(A)
        # reflection coefficient of the whole structure
        r = -res[1, 0] / res[0, 0]
        # transmission coefficient of the whole structure
        t = res[1, 1] - (res[1, 0] * res[0, 1]) / res[0, 0]
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(
            abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]])
        )

        return r, t, R, T


def diff_coefficient(struct, wavelength, incidence, polarization, method="S", pas=0.01):
    """
    This function computes the reflection and transmission coefficients derivative
    of the structure using the Transfer matrix formalism.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM
        method (string): T for T matrix, A for abeles matrix

    returns:
        dr (complex): deriv. of reflection coefficient, phase origin at first interface
        dt (complex): deriv. of transmission coefficient
        dR (float): deriv. of Reflectance (energy reflection)
        dT (float): deriv. of Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    """

    nb_var = len(struct.thickness) - 2 + len(struct.layer_type) - 2
    base_thickness = struct.thickness
    base_mat = np.array(struct.materials)
    base_lay = struct.layer_type

    dr = np.zeros(nb_var, dtype=complex)
    dt = np.zeros(nb_var, dtype=complex)
    dR = np.zeros(nb_var, dtype=float)
    dT = np.zeros(nb_var, dtype=float)

    function = None

    if method == "T":
        function = coefficient_with_grad_T
    elif method == "A":
        function = coefficient_with_grad_A

    if not function is None:
        # Using one of the fast differentiation methods
        r, t, R, T, A, B = function(
            struct, wavelength, incidence, polarization, mode="value"
        )

        for i in range(len(base_thickness) - 2):
            thickness = base_thickness.copy()
            thickness[i + 1] += pas
            struct = Structure(base_mat, base_lay, thickness, verbose=False)
            r_pas, t_pas, R_pas, T_pas = function(
                struct,
                wavelength,
                incidence,
                polarization,
                mode="grad",
                i_change=i + 1,
                saved_mat=[A, B],
            )
            dr[i] = (r_pas - r) / pas
            dt[i] = (t_pas - t) / pas
            dR[i] = (R_pas - R) / pas
            dT[i] = (T_pas - T) / pas

        for i in range(len(base_mat) - 2):
            mat = list(base_mat.copy())
            layer_type = base_lay.copy()

            mat.append(mat[base_lay[i + 1]].permittivity + pas)
            # Creating a new permittivity and referencing it
            layer_type[i + 1] = len(mat) - 1

            struct = Structure(mat, layer_type, base_thickness, verbose=False)
            r_pas, t_pas, R_pas, T_pas = function(
                struct,
                wavelength,
                incidence,
                polarization,
                mode="grad",
                i_change=i + 1,
                saved_mat=[A, B],
            )
            dr[i + len(base_thickness) - 2] = (r_pas - r) / pas
            dt[i + len(base_thickness) - 2] = (t_pas - t) / pas
            dR[i + len(base_thickness) - 2] = (R_pas - R) / pas
            dT[i + len(base_thickness) - 2] = (T_pas - T) / pas

        return dr, dt, dR, dT

    # Not using the fast computation -> using S matrix formalism
    function = coefficient_S
    (
        r,
        t,
        R,
        T,
    ) = function(struct, wavelength, incidence, polarization)

    for i in range(len(base_thickness) - 2):
        thickness = base_thickness.copy()
        thickness[i + 1] += pas
        struct = Structure(base_mat, base_lay, thickness, verbose=False)
        r_pas, t_pas, R_pas, T_pas = function(
            struct, wavelength, incidence, polarization
        )
        dr[i] = (r_pas - r) / pas
        dt[i] = (t_pas - t) / pas
        dR[i] = (R_pas - R) / pas
        dT[i] = (T_pas - T) / pas

    for i in range(len(base_mat) - 2):
        mat = list(base_mat.copy())
        layer_type = base_lay.copy()

        mat.append(mat[base_lay[i + 1]].permittivity + pas)
        # Creating a new permittivity and referencing it
        layer_type[i + 1] = len(mat) - 1

        struct = Structure(mat, layer_type, base_thickness, verbose=False)
        r_pas, t_pas, R_pas, T_pas = function(
            struct, wavelength, incidence, polarization
        )
        dr[i + len(base_thickness) - 2] = (r_pas - r) / pas
        dR[i + len(base_thickness) - 2] = (R_pas - R) / pas
        dt[i + len(base_thickness) - 2] = (t_pas - t) / pas
        dT[i + len(base_thickness) - 2] = (T_pas - T) / pas

    return dr, dt, dR, dT
