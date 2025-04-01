"""
This file contains all alternate methods to the S Matrix computation
"""

import numpy as np
from PyMoosh.classes import conv_to_nm
import copy


def coefficient_A(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the (true) Abeles matrix formalism.

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
    """
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

    T = np.zeros(((g - 1, 2, 2)), dtype=complex)
    c = np.cos(gamma * thickness)
    s = np.sin(gamma * thickness)
    gf = gamma / f[Type]
    for k in range(g - 1):
        # Layer scattering matrix

        T[k] = [[c[k], -s[k] / gf[k]], [gf[k] * s[k], c[k]]]
    # Once the scattering matrixes have been prepared, now let us combine them

    A = np.empty((2, 2), dtype=complex)
    A = T[0]
    for i in range(1, T.shape[0]):
        A = T[i] @ A

    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]

    amb = a - 1.0j * gf[0] * b
    apb = a + 1.0j * gf[0] * b
    cmd = c - 1.0j * gf[0] * d
    cpd = c + 1.0j * gf[0] * d
    # reflection coefficient of the whole structure

    r = -(cmd + 1.0j * gf[-1] * amb) / (cpd + 1.0j * gf[-1] * apb)
    # transmission coefficient of the whole structure
    t = a * (r + 1) + 1.0j * gf[0] * b * (r - 1)
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gf[g - 1] / gf[0])

    return r, t, R, T


def coefficient_T(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the Transfer matrix formalism.

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
    """
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

    T = np.zeros(((2 * g - 2, 2, 2)), dtype=complex)
    gf = gamma / f[Type]
    sum = (gf[1:] + gf[:-1]) / (2 * gf[1:])
    dif = (gf[1:] - gf[:-1]) / (2 * gf[1:])
    phases = np.exp(1.0j * gamma[1:] * thickness[1:])
    for k in range(g - 1):
        # Layer transfer matrix
        T[2 * k] = [[sum[k], dif[k]], [dif[k], sum[k]]]

        # Layer propagation matrix
        T[2 * k + 1] = [[1 / phases[k], 0], [0, phases[k]]]
    # Once the scattering matrixes have been prepared, now let us combine them

    A = np.empty((2, 2), dtype=complex)
    A = T[0]
    for i in range(1, T.shape[0]):
        A = A @ T[i]
    # reflection coefficient of the whole structure
    r = -A[1, 0] / A[0, 0]
    # transmission coefficient of the whole structure
    t = (A[1, 1] - (A[1, 0] * A[0, 1]) / A[0, 0]) * np.exp(
        -1j * gamma[-1] * thickness[-1]
    )
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gf[-1] / gf[0])

    return r, t, R, T


def cascade_DirtoNeu(A, B):
    """
    This function takes two 2x2 matrixes A and B, that are assumed to be Dirichlet to Neumann
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.

    Args:
        A (2x2 numpy array):
        B (2x2 numpy array):

    """
    t = 1 / (A[1, 1] - B[0, 0])
    S = np.zeros((2, 2), dtype=complex)
    S[0, 0] = A[0, 0] - A[0, 1] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = -B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + B[1, 0] * B[0, 1] * t
    return S


def coefficient_DN(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the Dirichlet to Neumann matrix formalism.

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
    """
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
    # of the first layer
    thickness[0] = wavelength / 100.0
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

    T = np.zeros(((g - 1, 2, 2)), dtype=complex)
    gf = gamma / f[Type]
    t = np.tan(gamma * thickness) / gf
    s = np.sin(gamma * thickness) / gf
    for k in range(g - 1):
        # Layer scattering matrix
        T[k] = np.array([[1 / t[k], -1 / s[k]], [1 / s[k], -1 / t[k]]])
    # Once the scattering matrixes have been prepared, now let us combine them

    A = np.empty(T.shape, dtype=complex)
    A[0] = T[0]
    for j in range(1, T.shape[0]):
        A[j] = cascade_DirtoNeu(A[j - 1], T[j])

    a = A[-1][0, 0]
    b = A[-1][0, 1]
    c = A[-1][1, 0]
    d = A[-1][1, 1]

    # reflection coefficient of the whole structure

    gamma_eps0 = 1.0j * gf[0]
    gamma_epsN = 1.0j * gf[-1]

    r = ((a + gamma_eps0) * (d + gamma_epsN) - b * c) / (
        (gamma_eps0 - a) * (d + gamma_epsN) + b * c
    )
    r = r * np.exp(-2j * gamma[0] * thickness[0])
    # transmission coefficient of the whole structure
    t = -2 * c * gamma_eps0 / ((gamma_eps0 - a) * (d + gamma_epsN) + b * c)
    t = t * np.exp(-1.0j * gamma[0] * thickness[0])
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gf[g - 1] / gf[0])

    return r, t, R, T


def coefficient_I(struct, wavelength, incidence, polarization):
    # n,d,lam,theta0):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the fast impedance formalism.

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
    """

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
    # Wavevector in vacuum.
    k0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g) * alpha**2)
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
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

    n_s = np.zeros(g, dtype=complex)

    n_layer = np.sqrt(Epsilon[Type] * Mu[Type])
    cos_theta = gamma / (k0 * n_layer)

    n_s = gamma / Mu[Type]
    opp = np.imag(n_s) > 0
    n_s = n_s - 2 * n_s * (opp)

    n_p = Epsilon[Type] / gamma
    opp = np.imag(n_p) > 0
    n_p = n_p - 2 * n_p * (opp)

    delta = np.array(2 * np.pi * thickness * n_layer * cos_theta / wavelength)
    temp = -1.0j * np.tan(delta)

    if polarization == 0:
        n_eff = n_s
    else:
        n_eff = n_p

    Y = n_eff[-1]

    PR = 1
    for m in np.arange(g - 2, -1, -1):
        Y = (Y + n_eff[m + 1] * temp[m + 1]) / (1 + Y * temp[m + 1] / n_eff[m + 1])
        if m > 0:
            PR *= np.cos(delta[m]) - 1.0j * Y * np.sin(delta[m]) / n_eff[m]

    r = (n_eff[0] - Y) / (n_eff[0] + Y)

    if polarization == 1:
        t = (n_eff[-1].real / n_eff[0].real) * (r + 1) / PR
    else:
        t = (1 + r) / PR

    if polarization == 1:
        r = -r
    R = abs(r) ** 2
    T = abs(t) ** 2
    return (r, t, R, T)


def absorption_A(struct, wavelength, incidence, polarization):
    """
    This function computes the percentage of the incoming energy
    that is absorbed in each layer when the structure is illuminated
    by a plane wave.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        absorb (numpy array): absorption in each layer
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)
    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.

    """
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

    T = np.zeros(((g - 1, 2, 2)), dtype=complex)
    c = np.cos(gamma * thickness)
    s = np.sin(gamma * thickness)
    gf = gamma / f[Type]
    for k in range(g - 1):
        # Layer scattering matrix

        T[k] = [[c[k], -s[k] / gf[k]], [gf[k] * s[k], c[k]]]
    # Once the scattering matrixes have been prepared, now let us combine them

    A = np.empty((T.shape[0], 2, 2), dtype=complex)
    A[0] = T[0]
    for i in range(1, T.shape[0]):
        A[i] = T[i] @ A[i - 1]

    a = A[-1][0, 0]
    b = A[-1][0, 1]
    c = A[-1][1, 0]
    d = A[-1][1, 1]

    amb = a - 1.0j * gf[0] * b
    apb = a + 1.0j * gf[0] * b
    cmd = c - 1.0j * gf[0] * d
    cpd = c + 1.0j * gf[0] * d
    # reflection coefficient of the whole structure

    r = -(cmd + 1.0j * gf[-1] * amb) / (cpd + 1.0j * gf[-1] * apb)
    # transmission coefficient of the whole structure
    t = a * (r + 1) + 1.0j * gf[0] * b * (r - 1)
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gf[g - 1] / gf[0])

    I = np.zeros(((A.shape[0] + 1, 2)), dtype=complex)

    for k in range(A.shape[0]):
        I[k, 0] = A[k][0, 0] * (r + 1) + A[k][0, 1] * (1.0j * gf[0] * (r - 1))
        I[k, 1] = A[k][1, 0] * (r + 1) + A[k][1, 1] * (1.0j * gf[0] * (r - 1))
        # Contains Ey and dzEy in layer k
    I[-1] = [t, -1.0j * gf[-1] * t]

    poynting = np.zeros(A.shape[0] + 1, dtype=complex)
    if polarization == 0:  # TE
        for k in range(A.shape[0] + 1):
            poynting[k] = np.real(-1.0j * I[k, 0] * np.conj(I[k, 1]) / gf[0])
    else:  # TM
        for k in range(A.shape[0] + 1):
            poynting[k] = np.real(1.0j * np.conj(I[k, 0]) * I[k, 1] / gf[0])
    # Absorption in each layer
    absorb = np.concatenate(([0], -np.diff(poynting)))
    # First layer is always supposed non absorbing

    return absorb, r, t, R, T
