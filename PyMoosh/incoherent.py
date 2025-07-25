import numpy as np
import copy
from PyMoosh.classes import conv_to_nm
from PyMoosh.core import cascade


def incoherent_coefficient_S(
    struct, incoherent_substrate, wavelength, incidence, polarization
):
    """
    This function computes the reflectance and transmittance coefficients
    of the structure, including an incoherent substrate.
    If incoherent_substrate is True, the last but one layer is considered an incoherent substrate

    Args:
        struct (Structure): belongs to the Structure class
        incoherent_substrate (boolean): whether the last but one layer is an incoherent substrate
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
    Incoherent warning: Only the last but one layer may be incoherent
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
    k_0 = 2 * np.pi / wavelength
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal
    alpha = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]]) * k_0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k_0 ** 2 - np.ones(g) * alpha ** 2)
    # Be cautious if the upper medium is a negative index one.
    if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
        gamma[0] = -gamma[0]

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 1] = gamma[1 : g - 1] * (1 - 2 * (np.imag(gamma[1 : g - 1]) < 0))
    # Outgoing wave condition for the last medium
    if (
        np.real(Epsilon[Type[g - 1]]) < 0
        and np.real(Mu[Type[g - 1]]) < 0
        and np.real(gamma[g - 1]) != 0
    ):
        gamma[g - 1] = -gamma[g - 1]
    # else:
    #     gamma[g - 1] = np.sqrt(
    #         Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k_0 ** 2 - alpha ** 2
    #     )
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

    if not (incoherent_substrate):
        # reflection coefficient of the whole structure
        r = A[len(A) - 1][0, 0]
        # transmission coefficient of the whole structure
        t = A[len(A) - 1][1, 0]
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(abs(t) ** 2 * gf[g - 1] / (gf[0]))

        return R, T
    else:
        # In the current version, the only incoherent layer is a substrate, which
        # is just before the external medium
        kz_sub = gamma[-2]
        loss_sub = np.exp(-2 * np.imag(kz_sub) * thickness[-2])

        # The matrix of the system from the top the the beginning of the substrate
        S = np.abs(A[-3]) ** 2

        n_sub = np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]] + 0j)
        n_out = np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]] + 0j)
        rs = np.abs((n_sub - n_out) / (n_sub + n_out)) ** 2  # normal incidence
        ts = np.abs((2 * n_sub) / (n_sub + n_out)) ** 2  # normal incidence

        C2 = S[1, 0] / (1 - S[1, 1] * loss_sub ** 2 * rs)
        R = S[0, 0] + S[0, 1] * C2 * rs * loss_sub ** 2
        T = C2 * loss_sub * ts * np.real(gf[g - 1] / (gf[0]))

        return R, T
