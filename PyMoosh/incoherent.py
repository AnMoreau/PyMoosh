import numpy as np
import copy
from PyMoosh.classes import conv_to_nm
from PyMoosh.core import cascade

# TODO: check indices for follow_growth
# TODO: check why follow growth works only with a semi infinite last layer


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
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2 - np.ones(g) * alpha**2)
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
        r = A[-1][0, 0]
        # transmission coefficient of the whole structure
        t = A[-1][1, 0]
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

        cos_theta_sub = kz_sub / (k_0 * np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]]))
        cos_theta_out = gamma[-1] / (k_0 * np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]]))
        n_sub = np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]] + 0j)
        n_out = np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]] + 0j)

        if polarization:  # TM
            denom = n_sub * cos_theta_out + n_out * cos_theta_sub
            rs = (n_out * cos_theta_sub - n_sub * cos_theta_out) / denom
            rs = np.abs(rs) ** 2
            ts = np.abs((2 * n_out * cos_theta_sub) / denom) ** 2
        else:  # TE
            n_cos_sub = n_sub * cos_theta_sub
            n_cos_out = n_out * cos_theta_out
            denom = n_cos_sub + n_cos_out
            rs = np.abs((n_cos_sub - n_cos_out) / denom) ** 2
            ts = np.abs((2 * n_cos_sub) / denom) ** 2

        C2 = S[1, 0] / (1 - S[1, 1] * loss_sub**2 * rs)
        R = S[0, 0] + S[0, 1] * C2 * rs * loss_sub**2
        T = C2 * loss_sub * ts * np.real(gf[g - 1] / (gf[0]))

        return R, T


def follow_growth_coefficient_S(
    struct,
    incoherent_substrate,
    wavelength,
    incidence,
    polarization,
    layer_change,
    prev_comp=None,
):
    """
    This function computes the reflectance and transmittance coefficients
    of the structure, including an incoherent substrate.
    If incoherent_substrate is True, the last but one layer is considered an incoherent substrate
    If prev_comp is given, it must contain information about all the S matrices computed for the structure
    with the previous parameters (typically, different height)

    Args:
        struct (Structure): belongs to the Structure class
        incoherent_substrate (boolean): whether the last but one layer is an incoherent substrate
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM
        layer_change (int): layer that will be changed
        prev_comp (list): [previous S matrices, layer height]

    returns:
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)
        S_mat (list(array)): S matrices above and below the changed layer


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning. (also, the last layer should be semi-infinite)
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

    if prev_comp is None:
        # Computation of the vertical wavevectors k_z
        gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2 - np.ones(g) * alpha**2)
        # Be cautious if the upper medium is a negative index one.
        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        # Changing the determination of the square root to achieve perfect stability
        if g > 2:
            gamma[1 : g - 1] = gamma[1 : g - 1] * (
                1 - 2 * (np.imag(gamma[1 : g - 1]) < 0)
            )
        # Outgoing wave condition for the last medium
        if (
            np.real(Epsilon[Type[g - 1]]) < 0
            and np.real(Mu[Type[g - 1]]) < 0
            and np.real(gamma[g - 1]) != 0
        ):
            gamma[g - 1] = -gamma[g - 1]
        gf = gamma / f[Type]

        T = np.zeros(((2 * g, 2, 2)), dtype=complex)

        # first S matrix
        T[0] = [[0, 1], [1, 0]]
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

        if not (incoherent_substrate):
            # Once the scattering matrixes have been prepared, now let us combine them

            H = np.zeros((len(T) - 1, 2, 2), dtype=complex)
            A = np.zeros((len(T) - 1, 2, 2), dtype=complex)

            H[0] = T[2 * g - 2]
            A[0] = T[0]

            for k in range(len(T) - 2):
                A[k + 1] = cascade(A[k], T[k + 1])
                H[k + 1] = cascade(T[2 * g - k - 3], H[k])

            # reflection coefficient of the whole structure
            r = A[-1][0, 0]
            # transmission coefficient of the whole structure
            t = A[-1][1, 0]
            # Energy reflexion coefficient;
            R = np.real(abs(r) ** 2)
            # Energy transmission coefficient;
            T = np.real(abs(t) ** 2 * gf[g - 1] / (gf[0]))

            return (
                R,
                T,
                [A[2 * layer_change], H[2 * g - (2 * layer_change + 4)]],
            )

        else:
            # In the current version, the only incoherent layer is a substrate, which
            # is just before the external medium
            # Once the scattering matrixes have been prepared, now let us combine them

            H = np.zeros((len(T) - 3, 2, 2), dtype=complex)
            A = np.zeros((len(T) - 3, 2, 2), dtype=complex)

            H[0] = T[2 * g - 4]
            A[0] = T[0]

            for k in range(len(T) - 4):
                A[k + 1] = cascade(A[k], T[k + 1])
                H[k + 1] = cascade(T[2 * g - k - 5], H[k])
            kz_sub = gamma[-2]
            loss_sub = np.abs(np.exp(-2 * np.imag(kz_sub) * thickness[-2]))

            # The matrix of the system from the top the the beginning of the substrate
            S = np.abs(A[-1]) ** 2

            cos_theta_sub = kz_sub / (k_0 * np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]]))
            cos_theta_out = gamma[-1] / (
                k_0 * np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]])
            )
            n_sub = np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]] + 0j)
            n_out = np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]] + 0j)

            if polarization:  # TM
                denom = n_sub * cos_theta_out + n_out * cos_theta_sub
                rs = (n_out * cos_theta_sub - n_sub * cos_theta_out) / denom
                rs = np.abs(rs) ** 2
                ts = np.abs((2 * n_out * cos_theta_sub) / denom) ** 2
            else:  # TE
                n_cos_sub = n_sub * cos_theta_sub
                n_cos_out = n_out * cos_theta_out
                denom = n_cos_sub + n_cos_out
                rs = np.abs((n_cos_sub - n_cos_out) / denom) ** 2
                ts = np.abs((2 * n_cos_sub) / denom) ** 2

            C2 = S[1, 0] / (1 - S[1, 1] * loss_sub**2 * rs)
            R = S[0, 0] + S[0, 1] * C2 * rs * loss_sub**2
            T = C2 * loss_sub * ts * np.real(gf[g - 1] / (gf[0]))

            return (
                R,
                T,
                [A[2 * layer_change], H[2 * g - (2 * layer_change + 6)]],
            )
    else:
        # prev_comp was given!

        # Computation of the vertical wavevectors k_z
        gamma_top = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]] * k_0**2 - alpha**2)
        gamma_bot = np.sqrt(Epsilon[Type[g - 1]] * Mu[Type[g - 1]] * k_0**2 - alpha**2)
        gamma_layer_change = np.sqrt(
            Epsilon[Type[layer_change]] * Mu[Type[layer_change]] * k_0**2 - alpha**2
        )
        # Be cautious if the upper medium is a negative index one.
        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma_top = -gamma_top
        # Outgoing wave condition for the last medium
        if (
            np.real(Epsilon[Type[g - 1]]) < 0
            and np.real(Mu[Type[g - 1]]) < 0
            and np.real(gamma_bot) != 0
        ):
            gamma_bot = -gamma_bot
        gf_top = gamma_top / f[Type[0]]
        gf_bot = gamma_bot / f[Type[g - 1]]

        [S_top, S_bot, new_h] = prev_comp
        t = np.exp((1j) * gamma_layer_change * new_h)
        T_new = np.array([[0, t], [t, 0]], dtype=complex)

        S = cascade(S_top, T_new)
        S = cascade(S, S_bot)

        if not incoherent_substrate:
            # reflection coefficient of the whole structure
            r = S[0, 0]
            # transmission coefficient of the whole structure
            t = S[1, 0]
            # Energy reflexion coefficient;
            R = np.real(abs(r) ** 2)
            # Energy transmission coefficient;
            T = np.real(abs(t) ** 2 * gf_bot / gf_top)

            return R, T, [S_top, S_bot]

        else:
            # incoherent substrate
            S = np.abs(
                S**2
            )  # TODO: If everything works as intended, this is all the way before the substrate

            kz_sub = np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]] * k_0**2 - alpha**2)
            loss_sub = np.abs(np.exp(-2 * np.imag(kz_sub) * thickness[-2]))
            cos_theta_sub = kz_sub / (k_0 * np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]]))
            cos_theta_out = np.sqrt(
                Epsilon[Type[-1]] * Mu[Type[-1]] * k_0**2 - alpha**2
            ) / (k_0 * np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]]))
            n_sub = np.sqrt(Epsilon[Type[-2]] * Mu[Type[-2]] + 0j)
            n_out = np.sqrt(Epsilon[Type[-1]] * Mu[Type[-1]] + 0j)

            if polarization:  # TM
                denom = n_sub * cos_theta_out + n_out * cos_theta_sub
                rs = (n_out * cos_theta_sub - n_sub * cos_theta_out) / denom
                rs = np.abs(rs) ** 2
                ts = np.abs((2 * n_out * cos_theta_sub) / denom) ** 2
            else:  # TE
                n_cos_sub = n_sub * cos_theta_sub
                n_cos_out = n_out * cos_theta_out
                denom = n_cos_sub + n_cos_out
                rs = np.abs((n_cos_sub - n_cos_out) / denom) ** 2
                ts = np.abs((2 * n_cos_sub) / denom) ** 2

            C2 = S[1, 0] / (1 - S[1, 1] * loss_sub**2 * rs)
            R = S[0, 0] + S[0, 1] * C2 * rs * loss_sub**2
            T = C2 * loss_sub * ts * np.real(gf_bot / (gf_top))

            return R, T, [S_top, S_bot]


def full_stack_incoherent(struct, wavelength, incidence, polarization):
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
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2 - np.ones(g) * alpha**2)
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
    T = np.abs(T) ** 2  # Switching to full incoherent mode
    A = np.zeros(((2 * g - 1, 2, 2)), dtype=float)
    A[0] = T[0]

    for j in range(len(T) - 2):
        A[j + 1] = cascade(A[j], T[j + 1])

    # reflection coefficient of the whole structure
    R = A[-1][0, 0]
    # transmission coefficient of the whole structure
    T = A[-1][1, 0] * np.real(gf[g - 1] / (gf[0]))

    return R, T
