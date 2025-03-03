"""
This file contains all functions linked to computing green functions
"""

from PyMoosh.core import cascade
from PyMoosh.classes import conv_to_nm
import numpy as np


def green(struct, window, lam, source_interface):
    """Computes the electric (TE polarization) field inside
    a multilayered structure illuminated by punctual source placed inside
    the structure.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        window (Window): description of the simulation domain
        lam (float): wavelength in vacuum
        source_interface (int): # of the interface where the source is located.
                                The medium should be the same on both sides.
    Returns:
        En (np.array): a matrix with the complex amplitude of the field

    Afterwards the matrix may be used to represent either the modulus or the
    real part of the field.
    """

    # Computation of all the permittivities/permeabilities
    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)
    Epsilon, Mu = struct.polarizability(lam)
    thickness = np.array(struct.thickness)
    pol = 0
    d = window.width
    C = window.C
    ny = np.floor(thickness / window.py)
    nx = window.nx
    Type = struct.layer_type
    print("Pixels vertically:", int(sum(ny)))

    # Check it's ready for the Green function :
    # Type of the layer is supposed to be the same
    # on both sides of the Interface

    if Type[source_interface - 1] != Type[source_interface]:
        print(
            "Error: there should be the same material on both sides "
            + "of the interface where the source is located."
        )
        return 0

    # Number of modes retained for the description of the field
    # so that the last mode has an amplitude < 1e-3 - you may want
    # to change it if the structure present reflexion coefficients
    # that are subject to very swift changes with the angle of incidence.

    # nmod = int(np.floor(0.83660 * d / w))
    nmod = 100

    # ----------- Do not touch this part ---------------
    l = lam / d
    thickness = thickness / d

    if pol == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum, no dimension
    k0 = 2 * np.pi / l
    # Initialization of the field component
    En = np.zeros((int(sum(ny)), int(nx)))
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1

    # Scattering matrix corresponding to no interface.
    T = np.zeros((2 * g + 2, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for nm in np.arange(2 * nmod + 1):
        # horizontal wavevector
        alpha = 2 * np.pi * (nm - nmod)
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

        Ampl = np.zeros((2 * g + 2, 2), dtype=complex)

        # -----------------------------> Above the source
        # calculation of the scattering matrices above the source (*_up)
        H_up = np.zeros((2 * source_interface, 2, 2), dtype=complex)
        A_up = np.zeros((2 * source_interface, 2, 2), dtype=complex)

        # T[2*source_interface] should be a neutral matrix for cascading
        # if the two media are the same on each side of the source.

        H_up[0] = [[0, 1], [1, 0]]
        A_up[0] = [[0, 1], [1, 0]]

        for k in range(2 * source_interface - 1):
            A_up[k + 1] = cascade(A_up[k], T[k + 1])
            H_up[k + 1] = cascade(T[2 * source_interface - 1 - k], H_up[k])

        I_up = np.zeros((2 * source_interface, 2, 2), dtype=complex)
        for k in range(2 * source_interface - 1):
            I_up[k] = np.array(
                [
                    [
                        A_up[k][1, 0],
                        A_up[k][1, 1] * H_up[2 * source_interface - 1 - k][0, 1],
                    ],
                    [
                        A_up[k][1, 0] * H_up[2 * source_interface - 1 - k][0, 0],
                        H_up[2 * source_interface - 1 - k][0, 1],
                    ],
                ]
                / (1 - A_up[k][1, 1] * H_up[2 * source_interface - 1 - k][0, 0])
            )

        # ----------------------------> Below the source
        # Calculation of the scattering matrices below the source (*_d)

        H_d = np.zeros((-2 * source_interface + 2 * g + 2, 2, 2), dtype=complex)
        A_d = np.zeros((-2 * source_interface + 2 * g + 2, 2, 2), dtype=complex)

        H_d[0] = [[0, 1], [1, 0]]
        A_d[0] = [[0, 1], [1, 0]]

        for k in range(2 * g + 1 - 2 * source_interface):
            A_d[k + 1] = cascade(A_d[k], T[2 * source_interface + k + 1])
            H_d[k + 1] = cascade(T[2 * g + 1 - k], H_d[k])

        I_d = np.zeros((-2 * source_interface + 2 * g + 2, 2, 2), dtype=complex)
        for k in range(2 * g + 1 - 2 * source_interface):
            I_d[k] = np.array(
                [
                    [
                        A_d[k][1, 0],
                        A_d[k][1, 1] * H_d[2 * (g - source_interface) + 1 - k][0, 1],
                    ],
                    [
                        A_d[k][1, 0] * H_d[2 * (g - source_interface) + 1 - k][0, 0],
                        H_d[2 * (g - source_interface) + 1 - k][0, 1],
                    ],
                ]
                / (1 - A_d[k][1, 1] * H_d[2 * (g - source_interface) + 1 - k][0, 0])
            )

        # >>> Inside the layer containing the source <<<

        r_up = A_up[2 * source_interface - 1][1, 1]
        r_d = H_d[2 * g + 1 - 2 * source_interface][0, 0]
        ex = -1j * np.exp(1j * alpha * window.C)
        # Multiply by -omega µ_0
        M = (
            ex
            / (1 - r_up + (1 + r_up) * (1 - r_d) / (1 + r_d))
            / gamma[source_interface]
        )
        D = (
            ex
            / (1 - r_d + (1 - r_up) / (1 + r_up) * (1 + r_d))
            / gamma[source_interface]
        )

        # Starting with the intermediary matrices, compute the right coefficients

        Ampl = np.zeros(2 * g + 2, dtype=complex)
        for k in range(source_interface):
            # Celui qui descend.
            Ampl[2 * k] = I_up[2 * k][0, 1] * M
            # Celui qui monte.
            Ampl[2 * k + 1] = I_up[2 * k + 1][1, 1] * M

        for k in range(source_interface, g + 1):
            Ampl[2 * k] = I_d[2 * (k - source_interface)][0, 0] * D
            Ampl[2 * k + 1] = I_d[2 * (k - source_interface) + 1][1, 0] * D

        Ampl[2 * source_interface - 1] = M
        Ampl[2 * source_interface] = D

        # >>> Calculation of the fields <<<

        h = 0
        t = 0
        E = np.zeros((int(np.sum(ny)), 1), dtype=complex)

        for k in range(g + 1):
            for m in range(int(ny[k])):
                h = h + float(thickness[k]) / ny[k]

                E[t, 0] = Ampl[2 * k] * np.exp(1j * gamma[k] * h) + Ampl[
                    2 * k + 1
                ] * np.exp(1j * gamma[k] * (thickness[k] - h))
                t += 1
            h = 0
        E = E * np.exp(1j * alpha * np.arange(0, nx) / nx)
        En = En + E

    return En


#    return r_up,r_d
