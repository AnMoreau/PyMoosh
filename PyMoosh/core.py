"""
This file contains the most fundamental functions for PyMoosh
Ideally, these would also be all the functions translated
into anisotropic and non local versions
"""

import numpy as np
from math import *
import copy
from PyMoosh.classes import conv_to_nm


def cascade(A, B):
    """
    This function takes two 2x2 matrixes A and B, that are assumed to be scattering matrixes
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.

    Args:
        A (2x2, 3x3, 4x4 numpy array):
        B (2x2, 3x3, 4x4 numpy array):

    """
    # If the interface or the layer is non-local, the matrix won't be size 2x2 so return the non-local cascade.
    # if np.shape(A) == (3,3) or (4,4) or np.shape(B) ==  (3,3) or (4,4) :
    #    return cascade_nl(A,B)
    t = 1 / (1 - B[0, 0] * A[1, 1])
    S = np.zeros((2, 2), dtype=complex)
    S[0, 0] = A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t
    return S

    if struct.NonLocal:
        print("Non Local field not yet defined")
    if struct.Anisotropic:
        print("Anisotropic field not yet defined")


def absorption(struct, wavelength, incidence, polarization):
    """
    This function computes the percentage of the incoming energy
    that is absorbed in each layer when the structure is illuminated
    by a plane wave.For now, uses the S matrix formalism

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in the same units as struct)
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
    return absorption_S(struct, wavelength, incidence, polarization)


def field(struct, beam, window):
    """Computes the electric (TE polarization) or magnetic (TM) field inside
    a multilayered structure illuminated by a gaussian beam.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        beam (Beam): description of the incidence beam
        window (Window): description of the simulation domain

    Returns:
        En (np.array): a matrix with the complex amplitude of the field

    Afterwards the matrix may be used to represent either the modulus or the
    real part of the field.
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

    # ----------- Do not touch this part ---------------
    l = lam / d
    w = w / d
    thickness = thickness / d

    if pol == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum, no dimension
    k0 = 2 * pi / l
    # Initialization of the field component
    En = np.zeros((int(sum(ny)), int(nx)))
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1
    # Amplitude of the different modes
    nmodvect = np.arange(-nmod, nmod + 1)
    # First factor makes the gaussian beam, the second one the shift
    # a constant phase is missing, it's just a change in the time origin.
    X = np.exp(-(w**2) * pi**2 * nmodvect**2) * np.exp(-2 * 1j * pi * nmodvect * C)

    # Scattering matrix corresponding to no interface.
    T = np.zeros((2 * g + 2, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for nm in np.arange(2 * nmod + 1):

        alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * sin(theta) + 2 * pi * (
            nm - nmod
        )
        gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g + 1) * alpha**2)

        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        if g > 2:
            gamma[1 : g - 1] = gamma[1 : g - 1] * (
                1 - 2 * (np.imag(gamma[1 : g - 1]) < 0)
            )
        if (
            np.real(Epsilon[Type[g]]) < 0
            and np.real(Mu[Type[g]]) < 0
            and np.real(np.sqrt(Epsilon[Type[g]] * k0**2 - alpha**2)) != 0
        ):
            gamma[g] = -np.sqrt(Epsilon[Type[g]] * Mu[Type[g]] * k0**2 - alpha**2)
        else:
            gamma[g] = np.sqrt(Epsilon[Type[g]] * Mu[Type[g]] * k0**2 - alpha**2)

        gf = gamma / f[Type]
        for k in range(g):
            t = np.exp(1j * gamma[k] * thickness[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])
            b1 = gf[k]
            b2 = gf[k + 1]
            T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)
        t = np.exp(1j * gamma[g] * thickness[g])
        T[2 * g + 1] = np.array([[0, t], [t, 0]])

        H = np.zeros((len(T) - 1, 2, 2), dtype=complex)
        A = np.zeros((len(T) - 1, 2, 2), dtype=complex)

        H[0] = T[2 * g + 1]
        A[0] = T[0]

        for k in range(len(T) - 2):
            A[k + 1] = cascade(A[k], T[k + 1])
            H[k + 1] = cascade(T[len(T) - k - 2], H[k])

        I = np.zeros((len(T), 2, 2), dtype=complex)
        for k in range(len(T) - 1):
            I[k] = np.array(
                [
                    [A[k][1, 0], A[k][1, 1] * H[len(T) - k - 2][0, 1]],
                    [A[k][1, 0] * H[len(T) - k - 2][0, 0], H[len(T) - k - 2][0, 1]],
                ]
                / (1 - A[k][1, 1] * H[len(T) - k - 2][0, 0])
            )

        h = 0
        t = 0

        E = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        for k in range(g + 1):
            for m in range(int(ny[k])):
                h = h + float(thickness[k]) / ny[k]
                # The expression for the field used here is based on the assumption
                # that the structure is illuminated from above only, with an Amplitude
                # of 1 for the incident wave. If you want only the reflected
                # field, take off the second term.
                E[t, 0] = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h) + I[2 * k + 1][
                    1, 0
                ] * np.exp(1j * gamma[k] * (thickness[k] - h))
                t += 1
            h = 0
        E = E * np.exp(1j * alpha * np.arange(0, nx) / nx)
        En = En + X[int(nm)] * E

    return En


def fields(struct, beam, window):
    """Computes the electric (TE polarization) or magnetic (TM) field inside
    a multilayered structure illuminated by a gaussian beam, and uses them
    to compute the rest of the fields (Hx,Hz in TE polarization/Ex,Ez in TM).
    To be precise, the derived fields need to be divided by omega*µ0 in TE
    and omega*epsilon_0 in TM. There a missing - sign in TM, too.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        beam (Beam): description of the incidence beam
        window (Window): description of the simulation domain

    Returns:
        En (np.array): a matrix with the complex amplitude of the field
        Hxn (np.array): a matrix with the complex amplitude of Hx/Ex
        Hzn (np.array): a matrix with the complex amplitude of Hz/Ez

    Afterwards the matrix may be used to represent either the modulus or the
    real part of the field.
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

    # ----------- Do not touch this part ---------------
    l = lam / d
    w = w / d
    thickness = thickness / d

    if pol == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum, no dimension
    k0 = 2 * pi / l
    # Initialization of the field component
    En = np.zeros((int(sum(ny)), int(nx)))
    Hxn = np.zeros((int(sum(ny)), int(nx)))
    Hzn = np.zeros((int(sum(ny)), int(nx)))
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1
    # Amplitude of the different modes
    nmodvect = np.arange(-nmod, nmod + 1)
    # First factor makes the gaussian beam, the second one the shift
    # a constant phase is missing, it's just a change in the time origin.
    X = np.exp(-(w**2) * pi**2 * nmodvect**2) * np.exp(-2 * 1j * pi * nmodvect * C)

    # Scattering matrix corresponding to no interface.
    T = np.zeros((2 * g + 2, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for nm in np.arange(2 * nmod + 1):

        alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k0 * sin(theta) + 2 * pi * (
            nm - nmod
        )
        gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g + 1) * alpha**2)

        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        if g > 2:
            gamma[1 : g - 1] = gamma[1 : g - 1] * (
                1 - 2 * (np.imag(gamma[1 : g - 1]) < 0)
            )
        if (
            np.real(Epsilon[Type[g]]) < 0
            and np.real(Mu[Type[g]]) < 0
            and np.real(np.sqrt(Epsilon[Type[g]] * k0**2 - alpha**2)) != 0
        ):
            gamma[g] = -np.sqrt(Epsilon[Type[g]] * Mu[Type[g]] * k0**2 - alpha**2)
        else:
            gamma[g] = np.sqrt(Epsilon[Type[g]] * Mu[Type[g]] * k0**2 - alpha**2)

        gf = gamma / f[Type]
        for k in range(g):
            t = np.exp(1j * gamma[k] * thickness[k])
            T[2 * k + 1] = np.array([[0, t], [t, 0]])
            b1 = gf[k]
            b2 = gf[k + 1]
            T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)
        t = np.exp(1j * gamma[g] * thickness[g])
        T[2 * g + 1] = np.array([[0, t], [t, 0]])

        H = np.zeros((len(T) - 1, 2, 2), dtype=complex)
        A = np.zeros((len(T) - 1, 2, 2), dtype=complex)

        H[0] = T[2 * g + 1]
        A[0] = T[0]

        for k in range(len(T) - 2):
            A[k + 1] = cascade(A[k], T[k + 1])
            H[k + 1] = cascade(T[len(T) - k - 2], H[k])

        I = np.zeros((len(T), 2, 2), dtype=complex)
        for k in range(len(T) - 1):
            I[k] = np.array(
                [
                    [A[k][1, 0], A[k][1, 1] * H[len(T) - k - 2][0, 1]],
                    [A[k][1, 0] * H[len(T) - k - 2][0, 0], H[len(T) - k - 2][0, 1]],
                ]
                / (1 - A[k][1, 1] * H[len(T) - k - 2][0, 0])
            )

        h = 0
        t = 0

        E = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        Hx = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        Hz = np.zeros((int(np.sum(ny)), 1), dtype=complex)
        for k in range(g + 1):
            for m in range(int(ny[k])):
                h = h + float(thickness[k]) / ny[k]
                # The expression for the field used here is based on the assumption
                # that the structure is illuminated from above only, with an Amplitude
                # of 1 for the incident wave. If you want only the reflected
                # field, take off the second term.
                E[t, 0] = I[2 * k][0, 0] * np.exp(1j * gamma[k] * h) + I[2 * k + 1][
                    1, 0
                ] * np.exp(1j * gamma[k] * (thickness[k] - h))
                Hx[t, 0] = -gf[k] * (
                    I[2 * k][0, 0] * np.exp(1j * gamma[k] * h)
                    - I[2 * k + 1][1, 0] * np.exp(1j * gamma[k] * (thickness[k] - h))
                )
                Hz[t, 0] = alpha * E[t, 0] / f[Type[k]]
                t += 1

            h = 0
        E = E * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Hx = Hx * np.exp(1j * alpha * np.arange(0, nx) / nx)
        Hz = Hz * np.exp(1j * alpha * np.arange(0, nx) / nx)

        En = En + X[int(nm)] * E
        Hxn = Hxn + X[int(nm)] * Hx
        Hzn = Hzn + X[int(nm)] * Hz

    return En, Hxn, Hzn


def coefficient(struct, wavelength, incidence, polarization):
    """
    Wrapper function to compute reflection and transmission coefficients
    with various methods.
    (and retrocompatibility)
    """
    return coefficient_S(struct, wavelength, incidence, polarization)


def coefficient_S(struct, wavelength, incidence, polarization):
    """
    This function computes the reflection and transmission coefficients
    of the structure.

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
    T = np.zeros(((2 * g, 2, 2)), dtype=complex)

    # first S matrix
    T[0] = [[0, 1], [1, 0]]
    gf = gamma / f[Type]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[k] * thickness[k])
        T[2 * k + 1] = [[0, t], [t, 0]]
        # Interface scattering matrix
        b1 = gf[k]
        b2 = gf[k + 1]
        T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)
    t = np.exp((1j) * gamma[g - 1] * thickness[g - 1])
    T[2 * g - 1] = [[0, t], [t, 0]]
    # Once the scattering matrixes have been prepared, now let us combine them
    A = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    A[0] = T[0]

    for j in range(len(T) - 2):
        A[j + 1] = cascade(A[j], T[j + 1])
    # reflection coefficient of the whole structure
    r = A[len(A) - 1][0, 0]
    # transmission coefficient of the whole structure
    t = A[len(A) - 1][1, 0]
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return r, t, R, T


def absorption_S(struct, wavelength, incidence, polarization, layers=[]):
    """
    This function computes the percentage of the incoming energy
    that is absorbed in each layer when the structure is illuminated
    by a plane wave.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM
        layers (list of int): which layers we must compute absorption in

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
    T = np.zeros(((2 * g, 2, 2)), dtype=complex)

    # first S matrix
    T[0] = [[0, 1], [1, 0]]
    gf = gamma / f[Type]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[k] * thickness[k])
        T[2 * k + 1] = [[0, t], [t, 0]]
        # Interface scattering matrix
        b1 = gf[k]
        b2 = gf[k + 1]
        T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]] / (b1 + b2))
    t = np.exp((1j) * gamma[g - 1] * thickness[g - 1])
    T[2 * g - 1] = [[0, t], [t, 0]]
    # Once the scattering matrixes have been prepared, now let us combine them
    D = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    U = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    D[0] = T[2 * g - 1]
    U[0] = T[0]
    for k in range(len(T) - 2):
        U[k + 1] = cascade(U[k], T[k + 1])
        D[k + 1] = cascade(T[2 * g - 2 - k], D[k])

    if len(layers) == 0:
        I = np.zeros(((2 * g, 2)), dtype=complex)
        for k in range(len(T) - 1):
            I[k] = (
                np.array([1, D[len(T) - k - 2][0, 0]], dtype=complex)
                * U[k][1, 0]
                / (1 - U[k][1, 1] * D[len(T) - k - 2][0, 0])
            )

        I[2 * g - 1][0] = I[2 * g - 2][0] * np.exp(1j * gamma[g - 1] * thickness[g - 1])
        I[2 * g - 1][1] = 0

        poynting = np.zeros(g, dtype=complex)
        if polarization == 0:  # TE
            for k in range(g):
                poynting[k] = np.real(
                    (I[2 * k + 1][0] + I[2 * k + 1][1])
                    * np.conj((I[2 * k + 1][0] - I[2 * k + 1][1]) * gf[k] / gf[0])
                )
        else:  # TM
            for k in range(g):
                poynting[k] = np.real(
                    (I[2 * k + 1][0] - I[2 * k + 1][1])
                    * np.conj((I[2 * k + 1][0] + I[2 * k + 1][1]))
                    * gf[k]
                    / gf[0]
                )
        # Absorption in each layer

        absorb = np.concatenate(([0], -np.diff(poynting)))
        # absorb=np.zeros(g,dtype=complex)
        # absorb = tmp[np.arange(0, 2 * g, 2)]

    else:
        # Specific layers are given for the absorption

        nb_lay = len(layers)
        layers = np.sort(layers)
        I = np.zeros(((2 * nb_lay, 2)), dtype=complex)
        i = 0
        for k in layers:
            if k != g - 1:
                I[2 * i] = (
                    np.array([1, D[len(T) - 2 * k - 2][0, 0]], dtype=complex)
                    * U[2 * k][1, 0]
                    / (1 - U[2 * k][1, 1] * D[len(T) - 2 * k - 2][0, 0])
                )
                I[2 * i + 1] = (
                    np.array([1, D[len(T) - 2 * k - 3][0, 0]], dtype=complex)
                    * U[2 * k + 1][1, 0]
                    / (1 - U[2 * k + 1][1, 1] * D[len(T) - 2 * k - 3][0, 0])
                )
                i += 1
        if g - 1 in layers:
            I[-2] = (
                np.array([1, D[len(T) - 2 * g][0, 0]], dtype=complex)
                * U[2 * g - 2][1, 0]
                / (1 - U[2 * g - 2][1, 1] * D[len(T) - 2 * g][0, 0])
            )
            I[-1][0] = I[-2][0] * np.exp(1j * gamma[g - 1] * thickness[g - 1])
            I[-1][1] = 0

        poynting = np.zeros(nb_lay, dtype=complex)
        if polarization == 0:  # TE
            for k in range(nb_lay):
                poynting[k] = np.real(
                    (I[2 * k + 1][0] + I[2 * k + 1][1])
                    * np.conj(
                        (I[2 * k + 1][0] - I[2 * k + 1][1]) * gf[layers[k]] / gf[0]
                    )
                )
        else:  # TM
            for k in range(nb_lay):
                poynting[k] = np.real(
                    (I[2 * k + 1][0] + I[2 * k + 1][1])
                    * np.conj((I[2 * k + 1][0] - I[2 * k + 1][1]) * gf[layers[k]])
                    / gf[0]
                )
        # Absorption in each layer

        absorb = -np.diff(poynting)
        if 0 in layers:
            absorb = np.concatenate(([0], absorb))
        # Remove abs value in case there is a material with gain
        # absorb = tmp[np.arange(0, 2 * nb_lay, 2)]
    # reflection coefficient of the whole structure
    r = U[len(U) - 1][0, 0]
    # transmission coefficient of the whole structure
    t = U[len(U) - 1][1, 0]
    # Energy reflexion coefficient;
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient;
    T = np.real(abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return absorb, r, t, R, T
