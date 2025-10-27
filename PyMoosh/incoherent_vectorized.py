import numpy as np
import copy
from PyMoosh.classes import conv_to_nm
from PyMoosh.core import cascade
from PyMoosh.vectorized import polarizability_opti_wavelength, cascade_opti

# TODO: check indices for follow_growth
# TODO: check why follow growth works only with a semi infinite last layer


def incoherent_spectrum_S(
    struct, incoherent_substrate, wavelengths, incidence, polarization
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
        wavelengths (list): the input wavelengths
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

    len_wl = len(wavelengths)
    len_mat = len(struct.materials)
    wavelengths.shape = (len_wl, 1)

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    if struct.unit != "nm":
        wavelengths = conv_to_nm(wavelengths, struct.unit)
    Epsilon, Mu = polarizability_opti_wavelength(struct, wavelengths)
    Epsilon.shape, Mu.shape = (len_wl, len_mat), (len_wl, len_mat)
    thickness = copy.deepcopy(struct.thickness)
    thickness = np.asarray(thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    thickness.shape = (1, len(thickness))
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k_0 = 2 * np.pi / wavelengths
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal Array of shape (len_wl, 1).
    Epsilon_first, Mu_first = Epsilon[:, Type[0]], Mu[:, Type[0]]
    Epsilon_first.shape, Mu_first.shape = (len_wl, 1), (len_wl, 1)
    alpha = np.sqrt(Epsilon_first * Mu_first) * k_0 * np.sin(incidence)

    # Computation of the vertical wavevectors k_z. Array of shape (len_wl, len_mat).
    gamma = np.sqrt(
        Epsilon[:, Type] * Mu[:, Type] * k_0**2 - np.ones((len_wl, g)) * alpha**2
    )

    # Be cautious if the upper medium is a negative index one.
    mask = np.logical_and(np.real(Epsilon_first) < 0, np.real(Mu_first) < 0)
    np.putmask(gamma[:, 0], mask, -gamma[:, 0])
    # Take - gamma if negative permittivity and permeability

    # Changing the determination of the square root to achieve perfect stability.
    if g > 2:
        gamma[:, 1 : g - 2] = gamma[:, 1 : g - 2] * (
            1 - 2 * (np.imag(gamma[:, 1 : g - 2]) < 0)
        )

    # Outgoing wave condition for the last medium.
    Epsilon_last, Mu_last = Epsilon[:, Type[g - 1]], Mu[:, Type[g - 1]]
    Epsilon_last.shape, Mu_last.shape = (len_wl, 1), (len_wl, 1)
    gamma_last = np.sqrt(Epsilon_last * Mu_last * k_0**2 - alpha**2)
    mask = np.logical_and.reduce(
        (np.real(Epsilon_last) < 0, np.real(Mu_last) < 0, np.real(gamma_last) != 0)
    )
    not_mask = np.logical_or.reduce(
        (np.real(Epsilon_last) > 0, np.real(Mu_last) > 0, np.real(gamma_last) == 0)
    )
    np.putmask(gamma[:, g - 1], mask, -gamma_last)
    np.putmask(gamma[:, g - 1], not_mask, gamma_last)
    # Take - gamma if negative permittivity and permeability

    T = np.zeros(((2 * g, 2, 2, len_wl, 1)), dtype=complex)

    # first S matrix
    zeros, ones = np.zeros((len_wl, 1)), np.ones((len_wl, 1))
    T[0] = [[zeros, ones], [ones, zeros]]

    gf = gamma / f[:, Type]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[:, k] * thickness[0, k])
        t.shape = (len_wl, 1)
        T[2 * k + 1] = [[zeros, t], [t, zeros]]
        # Interface scattering matrix
        b1 = gf[:, k]
        b2 = gf[:, k + 1]
        b1.shape, b2.shape = (len_wl, 1), (len_wl, 1)
        T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)

    t = np.exp((1j) * gamma[:, g - 1] * thickness[0, g - 1])
    t.shape = (len_wl, 1)
    T[2 * g - 1] = [[zeros, t], [t, zeros]]
    # Once the scattering matrixes have been prepared, now let us combine them
    A = np.zeros((2 * g - 1, 2, 2, len_wl, 1), dtype=complex)
    A[0] = T[0]

    for j in range(len(T) - 2):
        A[j + 1] = cascade_opti(A[j], T[j + 1], len_wl)

    if not (incoherent_substrate):
        # reflection coefficient of the whole structure
        r = A[-1][0, 0]
        # transmission coefficient of the whole structure
        t = A[-1][1, 0]
        # Energy reflexion coefficient;
        R = np.real(abs(r) ** 2)
        # Energy transmission coefficient;
        kz_coeff = gamma[:, g - 1] * f[:, Type[0]] / (gamma[:, 0] * f[:, Type[g - 1]])
        kz_coeff.shape = (len_wl, 1)
        T = abs(t) ** 2 * np.real(kz_coeff)
        return wavelengths, R, T
    else:
        # In the current version, the only incoherent layer is a substrate, which
        # is just before the external medium
        kz_sub = gamma[:, -2]
        loss_sub = np.exp(-2 * np.imag(kz_sub) * thickness[0, -2])

        # The matrix of the system from the top the the beginning of the substrate
        S = np.abs(A[-3]) ** 2
        cos_theta_sub = kz_sub / (
            k_0[:, 0] * np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]])
        )
        cos_theta_out = gamma[:, -1] / (
            k_0[:, 0] * np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]])
        )
        n_sub = np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]] + 0j)
        n_out = np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]] + 0j)

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

        C2 = S[1, 0][:, 0] / (1 - S[1, 1][:, 0] * loss_sub**2 * rs)
        R = S[0, 0][:, 0] + S[0, 1][:, 0] * C2 * rs * loss_sub**2
        kz_coeff = gamma[:, g - 1] * f[:, Type[0]] / (gamma[:, 0] * f[:, Type[g - 1]])
        T = C2 * loss_sub * ts * np.real(kz_coeff)
        return wavelengths, R, T


def follow_growth_spectrum_S(
    struct,
    incoherent_substrate,
    wavelengths,
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
        wavelengths (list): wavelengths of the incidence light (in nm)
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

    len_wl = len(wavelengths)
    len_mat = len(struct.materials)
    wavelengths.shape = (len_wl, 1)

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    if struct.unit != "nm":
        wavelengths = conv_to_nm(wavelengths, struct.unit)
    Epsilon, Mu = polarizability_opti_wavelength(struct, wavelengths)
    Epsilon.shape, Mu.shape = (len_wl, len_mat), (len_wl, len_mat)
    thickness = copy.deepcopy(struct.thickness)
    thickness = np.asarray(thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0
    thickness.shape = (1, len(thickness))
    Type = struct.layer_type
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Wavevector in vacuum.
    k_0 = 2 * np.pi / wavelengths
    # Number of layers
    g = len(struct.layer_type)
    # Wavevector k_x, horizontal Array of shape (len_wl, 1).
    Epsilon_first, Mu_first = Epsilon[:, Type[0]], Mu[:, Type[0]]
    Epsilon_first.shape, Mu_first.shape = (len_wl, 1), (len_wl, 1)
    alpha = np.sqrt(Epsilon_first * Mu_first) * k_0 * np.sin(incidence)

    if prev_comp is None:
        # Computation of the vertical wavevectors k_z. Array of shape (len_wl, len_mat).
        gamma = np.sqrt(
            Epsilon[:, Type] * Mu[:, Type] * k_0**2 - np.ones((len_wl, g)) * alpha**2
        )

        # Be cautious if the upper medium is a negative index one.
        mask = np.logical_and(np.real(Epsilon_first) < 0, np.real(Mu_first) < 0)
        np.putmask(gamma[:, 0], mask, -gamma[:, 0])
        # Take - gamma if negative permittivity and permeability

        # Changing the determination of the square root to achieve perfect stability.
        if g > 2:
            gamma[:, 1 : g - 2] = gamma[:, 1 : g - 2] * (
                1 - 2 * (np.imag(gamma[:, 1 : g - 2]) < 0)
            )

        # Outgoing wave condition for the last medium.
        Epsilon_last, Mu_last = Epsilon[:, Type[g - 1]], Mu[:, Type[g - 1]]
        Epsilon_last.shape, Mu_last.shape = (len_wl, 1), (len_wl, 1)
        gamma_last = np.sqrt(Epsilon_last * Mu_last * k_0**2 - alpha**2)
        mask = np.logical_and.reduce(
            (np.real(Epsilon_last) < 0, np.real(Mu_last) < 0, np.real(gamma_last) != 0)
        )
        not_mask = np.logical_or.reduce(
            (np.real(Epsilon_last) > 0, np.real(Mu_last) > 0, np.real(gamma_last) == 0)
        )
        np.putmask(gamma[:, g - 1], mask, -gamma_last)
        np.putmask(gamma[:, g - 1], not_mask, gamma_last)
        # Take - gamma if negative permittivity and permeability

        T = np.zeros(((2 * g, 2, 2, len_wl, 1)), dtype=complex)

        # first S matrix
        zeros, ones = np.zeros((len_wl, 1)), np.ones((len_wl, 1))
        T[0] = [[zeros, ones], [ones, zeros]]

        gf = gamma / f[:, Type]
        for k in range(g - 1):
            # Layer scattering matrix
            t = np.exp((1j) * gamma[:, k] * thickness[0, k])
            t.shape = (len_wl, 1)
            T[2 * k + 1] = [[zeros, t], [t, zeros]]
            # Interface scattering matrix
            b1 = gf[:, k]
            b2 = gf[:, k + 1]
            b1.shape, b2.shape = (len_wl, 1), (len_wl, 1)
            T[2 * k + 2] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)

        t = np.exp((1j) * gamma[:, g - 1] * thickness[0, g - 1])
        t.shape = (len_wl, 1)
        T[2 * g - 1] = [[zeros, t], [t, zeros]]

        if not (incoherent_substrate):
            # Once the scattering matrixes have been prepared, now let us combine them

            H = np.zeros((2 * g - 1, 2, 2, len_wl, 1), dtype=complex)
            A = np.zeros((2 * g - 1, 2, 2, len_wl, 1), dtype=complex)

            H[0] = T[2 * g - 2]
            A[0] = T[0]

            for k in range(len(T) - 2):
                A[k + 1] = cascade_opti(A[k], T[k + 1], len_wl)
                H[k + 1] = cascade_opti(T[2 * g - k - 3], H[k], len_wl)

            # reflection coefficient of the whole structure
            r = A[-1][0, 0]
            # transmission coefficient of the whole structure
            t = A[-1][1, 0]
            # Energy reflexion coefficient;
            R = np.real(abs(r) ** 2)
            # Energy transmission coefficient;
            kz_coeff = (
                gamma[:, g - 1] * f[:, Type[0]] / (gamma[:, 0] * f[:, Type[g - 1]])
            )
            kz_coeff.shape = (len_wl, 1)
            T = abs(t) ** 2 * np.real(kz_coeff)

            return (
                wavelengths,
                R,
                T,
                [A[2 * layer_change], H[2 * g - (2 * layer_change + 4)]],
            )

        else:
            # In the current version, the only incoherent layer is a substrate, which
            # is just before the external medium
            # Once the scattering matrixes have been prepared, now let us combine them

            H = np.zeros((len(T) - 3, 2, 2, len_wl, 1), dtype=complex)
            A = np.zeros((len(T) - 3, 2, 2, len_wl, 1), dtype=complex)

            H[0] = T[2 * g - 4]
            A[0] = T[0]

            for k in range(len(T) - 4):
                A[k + 1] = cascade_opti(A[k], T[k + 1], len_wl)
                H[k + 1] = cascade_opti(T[2 * g - k - 5], H[k], len_wl)
            kz_sub = gamma[:, -2]
            loss_sub = np.abs(np.exp(-2 * np.imag(kz_sub) * thickness[0, -2]))

            # The matrix of the system from the top the the beginning of the substrate
            S = np.abs(A[-1]) ** 2

            cos_theta_sub = kz_sub / (
                k_0[:, 0] * np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]])
            )
            cos_theta_out = gamma[:, -1] / (
                k_0[:, 0] * np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]])
            )
            n_sub = np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]] + 0j)
            n_out = np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]] + 0j)

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

            C2 = S[1, 0][:, 0] / (1 - S[1, 1][:, 0] * loss_sub**2 * rs)
            R = S[0, 0][:, 0] + S[0, 1][:, 0] * C2 * rs * loss_sub**2
            kz_coeff = (
                gamma[:, g - 1] * f[:, Type[0]] / (gamma[:, 0] * f[:, Type[g - 1]])
            )
            T = C2 * loss_sub * ts * np.real(kz_coeff)

            return (
                wavelengths,
                R,
                T,
                [A[2 * layer_change], H[2 * g - (2 * layer_change + 6)]],
            )
    else:
        # prev_comp was given!

        # Computation of the vertical wavevectors k_z
        gamma_top = np.sqrt(
            Epsilon[:, Type[0]] * Mu[:, Type[0]] * k_0[:, 0] ** 2 - alpha[:, 0] ** 2
        )
        gamma_bot = np.sqrt(
            Epsilon[:, Type[g - 1]] * Mu[:, Type[g - 1]] * k_0[:, 0] ** 2
            - alpha[:, 0] ** 2
        )
        gamma_layer_change = np.sqrt(
            Epsilon[:, Type[layer_change]] * Mu[:, Type[layer_change]] * k_0[:, 0] ** 2
            - alpha[:, 0] ** 2
        )
        # TODO: adapt to dispersive first medium
        # Be cautious if the upper medium is a negative index one.
        if np.real(Epsilon[0, Type[0]]) < 0 and np.real(Mu[0, Type[0]]) < 0:
            gamma_top = -gamma_top
        # Outgoing wave condition for the last medium
        if (
            np.real(Epsilon[0, Type[g - 1]]) < 0
            and np.real(Mu[0, Type[g - 1]]) < 0
            and np.real(gamma_bot) != 0
        ):
            gamma_bot = -gamma_bot
        gf_top = gamma_top / f[:, Type[0]]
        gf_bot = gamma_bot / f[:, Type[g - 1]]

        [S_top, S_bot, new_h] = prev_comp
        t = np.exp((1j) * gamma_layer_change * new_h)
        zeros, ones = np.zeros((len_wl, 1)), np.ones((len_wl, 1))
        t.shape = (len_wl, 1)
        T_new = np.array([[zeros, t], [t, zeros]], dtype=complex)

        S = cascade_opti(S_top, T_new, len_wl)
        S = cascade_opti(S, S_bot, len_wl)

        if not incoherent_substrate:
            # reflection coefficient of the whole structure
            r = S[0, 0]
            # transmission coefficient of the whole structure
            t = S[1, 0]
            # Energy reflexion coefficient;
            R = np.real(abs(r) ** 2)
            # Energy transmission coefficient;
            kz_coeff = gf_bot / gf_top
            kz_coeff.shape = (len_wl, 1)
            T = abs(t) ** 2 * np.real(kz_coeff)

            return wavelengths, R, T, [S_top, S_bot]

        else:
            # incoherent substrate
            S = np.abs(S**2)

            kz_sub = np.sqrt(
                Epsilon[:, Type[-2]] * Mu[:, Type[-2]] * k_0[:, 0] ** 2
                - alpha[:, 0] ** 2
            )
            loss_sub = np.abs(np.exp(-2 * np.imag(kz_sub) * thickness[0, -2]))
            cos_theta_sub = kz_sub / (
                k_0[:, 0] * np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]])
            )
            cos_theta_out = np.sqrt(
                Epsilon[:, Type[-1]] * Mu[:, Type[-1]] * k_0[:, 0] ** 2
                - alpha[:, 0] ** 2
            ) / (k_0[:, 0] * np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]]))
            n_sub = np.sqrt(Epsilon[:, Type[-2]] * Mu[:, Type[-2]] + 0j)
            n_out = np.sqrt(Epsilon[:, Type[-1]] * Mu[:, Type[-1]] + 0j)

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

            C2 = S[1, 0][:, 0] / (1 - S[1, 1][:, 0] * loss_sub**2 * rs)
            R = S[0, 0][:, 0] + S[0, 1][:, 0] * C2 * rs * loss_sub**2
            kz_coeff = gf_bot / gf_top
            T = C2 * loss_sub * ts * np.real(kz_coeff)

            return wavelengths, R, T, [S_top, S_bot]


def full_stack_incoherent(struct, wavelength, incidence, polarization):
    """
    This function computes the reflectance and transmittance coefficients
    of the structure, including an incoherent substrate.
    If incoherent_substrate is True, the last but one layer is considered an incoherent substrate

    Args:
        struct (Structure): belongs to the Structure class
        incoherent_substrate (boolean): whether the last but one layer is an incoherent substrate
        wavelengths (list): wavelengths of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        wavelengths (list): the input wavelengths
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
