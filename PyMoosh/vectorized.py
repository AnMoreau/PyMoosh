"""
This file contains all functions containing loops over a given variable
that are optimized using numpy
"""
# TODO: check polarizability opti works for all material types
# TODO: check absorption for spectrum_A works
# TODO: parallelize absorption for spectrum_S
# TODO: add absorb keyword in the wrapper
# TODO: angular (A, S, absorption for both)
import numpy as np
import sys
print(sys.path)
from PyMoosh.core import coefficient
import copy
from PyMoosh.classes import conv_to_nm


def polarizability_opti_wavelength(struct, wavelengths): #numpy friendly
    """ ** Only used for coefficient_S_opti_wavelength **
    Computes the actual permittivity and permeability of each material considered in
    the structure. This method is called before each calculation.

    Args:
        wavelengths (numpy array): the working wavelengths (in nanometers)

    TODO: check if this works for any type of material
    """


    # Create empty mu and epsilon arrays
    mu = np.ones((wavelengths.size, len(struct.materials)), dtype=np.clongdouble)
    epsilon = np.ones((wavelengths.size, len(struct.materials)), dtype=np.clongdouble)
    # Loop over all materials
    for k in range(len(struct.materials)):
        # Populate epsilon and mu arrays from the material.
        material = struct.materials[k]
        material_get_permittivity = material.get_permittivity(wavelengths)
        material_get_permeability = material.get_permeability(wavelengths)
        try:
            material_get_permittivity.shape = (len(wavelengths),)
            material_get_permeability.shape = (len(wavelengths),)
        except:
            pass

        epsilon[:,k] = material_get_permittivity
        mu[:,k] = material_get_permeability

    return epsilon, mu




def cascade_opti_wavelength(A, B, len_wl): #numpy friendly
    """
    ** Only used in coefficient_S_opti_wavelength definition **

    This function takes two 2x2 matrixes A and B of (len_wl, 1) arrays, that are assumed to be scattering matrixes
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix of (len_wl, 1) arrays.

    Args:
        A (2x2 numpy array of (len_wl, 1) arrays):
        B (2x2 numpy arrayof (len_wl, 1) arrays):

    """
    t = 1 / (1 - B[0, 0] * A[1, 1])
    S = np.zeros((2, 2, len_wl, 1), dtype=complex)
    S[0, 0] = A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t
    return (S)


def spectrum(struct, incidence, polarization, wl_min, wl_max, len_wl, method="S"):
    """
        Wrapper to choose between S matrices (stability) and Abélès (stability)
    """
    if (method == "S"):
        return spectrum_S(struct, incidence, polarization, wl_min, wl_max, len_wl)
    elif (method == "A"):
        return spectrum_A(struct, incidence, polarization, wl_min, wl_max, len_wl)

def spectrum_S(struct, incidence, polarization, wl_min, wl_max, len_wl):
    """
    Represents the reflexion coefficient (reflectance and phase) for a
    multilayered structure. This is an vectorized version of the :coefficient:
    function over an array of wavelengths.

    Args:
        structure (Structure): the object describing the multilayer
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 for TM
        wl_min (float): minimum wavelength of the spectrum
        wl_max (float): maximum wavelength of the spectrum
        len_wl (int): number of points in the spectrum

    Returns:
        wavelengths (numpy array): wavelength considered
        r (numpy complex array): reflexion coefficient for each wavelength
        t (numpy complex array): transmission coefficient
        R (numpy array): Reflectance
        T (numpy array): Transmittance


    """
    
    # In order to get a phase that corresponds to the expected reflected coefficient,
    # we make the height of the upper (lossless) medium vanish. It changes only the
    # phase of the reflection coefficient.

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    wavelengths = np.linspace(wl_min, wl_max, len_wl)
    len_mat = len(struct.materials)
    wavelengths.shape = (len_wl, 1)

    if (struct.unit != "nm"):
        wavelengths = conv_to_nm(wavelengths, struct.unit)

    # Epsilon and Mu are (len_wl, len_mat) arrays.
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

    # Wavevector in vacuum. Array of shape (len_wl, 1).
    k0 = 2 * np.pi / wavelengths

    # Number of layers
    g = len(struct.layer_type)

    # Wavevector k_x, horizontal. Array of shape (len_wl, 1).
    Epsilon_first, Mu_first = Epsilon[:,Type[0]], Mu[:,Type[0]]
    Epsilon_first.shape, Mu_first.shape = (len_wl, 1), (len_wl, 1)
    alpha = np.sqrt(Epsilon_first * Mu_first) * k0 * np.sin(incidence)

    # Computation of the vertical wavevectors k_z. Array of shape (len_wl, len_mat).
    gamma = np.sqrt(
        Epsilon[:,Type] * Mu[:,Type] * k0 ** 2 - np.ones((len_wl, g)) * alpha ** 2)

    # Be cautious if the upper medium is a negative index one.
    mask = np.logical_and(np.real(Epsilon_first) < 0, np.real(Mu_first) < 0)
    np.putmask(gamma[:,0], mask,-gamma[:,0])
    #TODO: check what this does

    # Changing the determination of the square root to achieve perfect stability.
    if g > 2:
        gamma[:,1:g - 2] = gamma[:,1:g - 2] * (1 - 2 * (np.imag(gamma[:,1:g - 2]) < 0))

    # Outgoing wave condition for the last medium.
    Epsilon_last, Mu_last = Epsilon[:,Type[g-1]], Mu[:,Type[g-1]]
    Epsilon_last.shape, Mu_last.shape = (len_wl, 1), (len_wl, 1)
    gamma_last = np.sqrt(Epsilon_last* Mu_last * k0 ** 2 - alpha ** 2)
    mask = np.logical_and.reduce(
    (np.real(Epsilon_last) < 0, np.real(Mu_last) < 0, np.real(gamma_last) != 0))
    not_mask = np.logical_or.reduce(
    (np.real(Epsilon_last) > 0, np.real(Mu_last) > 0, np.real(gamma_last) == 0))
    np.putmask(gamma[:,g-1], mask, -gamma_last)
    np.putmask(gamma[:,g-1], not_mask, gamma_last)
    #TODO: check what this does

    # Each layer has a (2, 2) matrix with (len_wl, 1) array as coefficient.
    T = np.zeros(((2 * g, 2, 2, len_wl, 1)), dtype=complex)

    # first S matrix
    zeros, ones = np.zeros((len_wl, 1)), np.ones((len_wl, 1))
    T[0] = [[zeros, ones], [ones, zeros]]
    gf = gamma / f[:,Type]
    for k in range(g - 1):
        # Layer scattering matrix
        t = np.exp((1j) * gamma[:,k] * thickness[0,k])
        t.shape = (len_wl, 1)
        T[2 * k + 1] = [[zeros, t], [t, zeros]]

        # Interface scattering matrix
        b1 = gf[:,k]
        b2 = gf[:,k + 1]
        b1.shape, b2.shape = (len_wl, 1), (len_wl, 1)
        T[2 * k + 2] = np.array([[b1 - b2, 2 * b2],
                                 [2 * b1, b2 - b1]]) / (b1 + b2)

    t = np.exp((1j) * gamma[:,g - 1] * thickness[0,g - 1])
    t.shape = (len_wl, 1)
    T[2 * g - 1] = [[zeros, t], [t, zeros]]

    # Once the scattering matrixes have been prepared, now let us combine them
    A = np.zeros(((2 * g - 1, 2, 2, len_wl, 1)), dtype=complex)
    A[0] = T[0]

    for j in range(len(T) - 2):
        A[j + 1] = cascade_opti_wavelength(A[j], T[j + 1], len_wl)
    # reflection coefficient of the whole structure
    r = A[len(A) - 1][0, 0]
    # transmission coefficient of the whole structure
    t = A[len(A) - 1][1, 0]
    # Energy reflexion coefficient;
    R = np.real(np.absolute(r) ** 2)
    # Energy transmission coefficient;
    temp = gamma[:,g - 1] * f[:,Type[0]] / (gamma[:,0] * f[:,Type[g - 1]])
    temp.shape = (len_wl, 1)
    T = np.absolute(t) ** 2 * np.real(temp)

    return wavelengths, r, t, R, T


def spectrum_A(struct, incidence, polarization, wl_min, wl_max, len_wl, absorb=False):
    """
    This function computes the reflection and transmission coefficients
    of the structure using the (true) Abeles matrix formalism.
    If absorb is set to True, also returns the absorption in each layer

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        wavelengths (numpy array): wavelengths considered
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)
        if absorb is True, returns A (float): the absorption in each layer


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    """
    # In order to get a phase that corresponds to the expected reflected coefficient,
    # we make the height of the upper (lossless) medium vanish. It changes only the
    # phase of the reflection coefficient.

    # The medium may be dispersive. The permittivity and permability of each
    # layer has to be computed each time.
    wavelengths = np.linspace(wl_min, wl_max, len_wl)
    len_mat = len(struct.materials)
    wavelengths.shape = (len_wl, 1)

    if (struct.unit != "nm"):
        wavelengths = conv_to_nm(wavelengths, struct.unit)

    # Epsilon and Mu are (len_wl, len_mat) arrays.
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

    # Wavevector in vacuum. Array of shape (len_wl, 1).
    k0 = 2 * np.pi / wavelengths

    # Number of layers.
    g = len(struct.layer_type)

    # Wavevector k_x, horizontal. Array of shape (len_wl, 1).
    Epsilon_first, Mu_first = Epsilon[:,Type[0]], Mu[:,Type[0]]
    Epsilon_first.shape, Mu_first.shape = (len_wl, 1), (len_wl, 1)
    alpha = np.sqrt(Epsilon_first * Mu_first) * k0 * np.sin(incidence)
    # Computation of the vertical wavevectors k_z. Array of shape (len_wl, len_mat).
    gamma = np.sqrt(
        Epsilon[:,Type] * Mu[:,Type] * k0 ** 2 - np.ones((len_wl, g)) * alpha ** 2)

    # Be cautious if the upper medium is a negative index one.
    mask = np.logical_and(np.real(Epsilon_first) < 0, np.real(Mu_first) < 0)
    np.putmask(gamma[:,0], mask,-gamma[:,0])

    # Changing the determination of the square root to achieve perfect stability.
    if g > 2:
        gamma[:,1:g - 2] = gamma[:,1:g - 2] * (1 - 2 * (np.imag(gamma[:,1:g - 2]) < 0))

    # Outgoing wave condition for the last medium.
    Epsilon_last, Mu_last = Epsilon[:,Type[g-1]], Mu[:,Type[g-1]]
    Epsilon_last.shape, Mu_last.shape = (len_wl, 1), (len_wl, 1)
    gamma_last = np.sqrt(Epsilon_last* Mu_last * k0 ** 2 - alpha ** 2)
    mask = np.logical_and.reduce(
    (np.real(Epsilon_last) < 0, np.real(Mu_last) < 0, np.real(gamma_last) != 0))
    not_mask = np.logical_or.reduce(
    (np.real(Epsilon_last) > 0, np.real(Mu_last) > 0, np.real(gamma_last) == 0))
    np.putmask(gamma[:,g-1], mask, -gamma_last)
    np.putmask(gamma[:,g-1], not_mask, gamma_last)

    # Each layer has a (2, 2) matrix with (len_wl, 1) array as coefficient.
    T = np.zeros(((g-1, 2, 2, len_wl)), dtype=np.clongdouble)
    c = np.cos(gamma * thickness)
    s = np.sin(gamma * thickness)
    gf = gamma/f[:,Type]

    for k in range(g-1):
        # Layer scattering matrix
        c_k, s_k, gf_k = c[:,k], s[:,k], gf[:,k]
        c_k.shape, s_k.shape, gf_k.shape = (len_wl), (len_wl), (len_wl)
        T[k] = np.array([[c_k, -s_k / gf_k],
                         [gf_k * s_k, c_k]])

    # Once the scattering matrixes have been prepared, now let us combine them

    if not(absorb):
        A = np.empty((2, 2, len_wl), dtype=np.clongdouble)
        A = T[0]

        # We change the form of the matrix A to use numpy methods.
        for i in range(1, T.shape[0]):
            B = T[i,:,:,:]
            A = np.transpose(A, (2,0,1))
            B = np.transpose(B, (2,0,1))
            A = np.matmul(B, A)
            A = np.transpose(A, (1,2,0))

        a = A[:][0, 0]
        b = A[:][0, 1]
        c = A[:][1, 0]
        d = A[:][1, 1]

        amb = a - 1.j * gf[:,0] * b
        apb = a + 1.j * gf[:,0] * b
        cmd = c - 1.j * gf[:,0] * d
        cpd = c + 1.j * gf[:,0] * d

        # reflection coefficient of the whole structure
        r = -(cmd + 1.j * gf[:,-1] * amb)/(cpd + 1.j * gf[:,-1] * apb)
        # transmission coefficient of the whole structure
        t = a * (r+1) + 1.j * gf[:,0] * b * (r-1)
        # Energy reflexion coefficient;
        R = np.real(np.absolute(r) ** 2)
        # Energy transmission coefficient;
        T = np.absolute(t) ** 2 * np.real(gf[:,g - 1] / gf[:,0])

        return wavelengths, r, t, R, T
    
    if (absorb):
        # Compute absorption in addition to r, t, R, T
        A = np.empty((T.shape[0], 2, 2, len_wl), dtype=np.clongdouble)
        A[0] = T[0]

        # We change the form of the matrix A to use numpy methods.
        for i in range(1, T.shape[0]):
            Y = np.transpose(A[i-1], (2,0,1))
            X = np.transpose(T[i], (2,0,1))
            Z = np.matmul(X, Y)
            A[i] = np.transpose(Z, (1,2,0))

        a = A[-1][0, 0][:]
        b = A[-1][0, 1][:]
        c = A[-1][1, 0][:]
        d = A[-1][1, 1][:]

        amb = a - 1.j * gf[:,0] * b
        apb = a + 1.j * gf[:,0] * b
        cmd = c - 1.j * gf[:,0] * d
        cpd = c + 1.j * gf[:,0] * d

        # reflection coefficient of the whole structure
        r = -(cmd + 1.j * gf[:,-1] * amb)/(cpd + 1.j * gf[:,-1] * apb)
        # transmission coefficient of the whole structure
        t = a * (r+1) + 1.j * gf[:,0] * b * (r-1)
        # Energy reflexion coefficient;
        R = np.real(np.absolute(r) ** 2)
        # Energy transmission coefficient;
        T = np.real(np.absolute(t) ** 2 * gf[:,g - 1] / gf[:,0])


        I = np.zeros(((A.shape[0]+1, 2, len_wl)), dtype=complex)

        for k in range(A.shape[0]):
            I[k,0][:] = A[k][0, 0][:] * (r + np.ones_like(r)) + A[k][0, 1][:]*(1.j*gf[:,0]*(r - np.ones_like(r)))
            I[k,1][:] = A[k][1, 0][:] * (r + np.ones_like(r)) + A[k][1, 1][:]*(1.j*gf[:,0]*(r - np.ones_like(r)))
            # Contains Ey and dzEy in layer k
        I[-1,:] = [t, -1.j*gf[:,-1]*t]

        poynting = np.zeros((A.shape[0]+1, len_wl), dtype=complex)
        if polarization == 0:  # TE
            for k in range(A.shape[0]+1):
                poynting[k,:] = np.real(-1.j * I[k, 0,:] * np.conj(I[k, 1,:]) / gf[:,0])
        else:  # TM
            for k in range(A.shape[0]+1):
                poynting[k,:] = np.real(1.j * np.conj(I[k, 0,:]) * I[k, 1,:] / gf[:,0])
        # Absorption in each layer

        zeros = np.zeros((1, len_wl))
        diff_poynting = abs(-np.diff(poynting, axis=0))
        absorb = np.concatenate((zeros, diff_poynting), axis=0)
        absorb = np.transpose(absorb)
        # First layer is always supposed non absorbing

        return wavelengths, r, t, R, T, absorb



def angular(structure, wavelength, polarization, theta_min, theta_max,
            n_points):
    """Represents the reflexion coefficient (reflectance and phase) for a
    multilayered structure. This is an automated call to the :coefficient:
    function making the angle of incidence vary.

    Args:
        structure (Structure): the object describing the multilayer
        wavelength (float): the working wavelength in nm
        polarization (float): 0 for TE, 1 for TM
        theta_min (float): minimum angle of incidence in degrees
        theta_max (float): maximum angle of incidence in degrees
        n_points (int): number of different angle of incidence

    Returns:
        incidence (numpy array): angles of incidence considered
        r (numpy complex array): reflexion coefficient for each angle
        t (numpy complex array): transmission coefficient
        R (numpy array): Reflectance
        T (numpy array): Transmittance

    .. warning: The incidence angle is in degrees here, contrarily to
    other functions.

    """
    # theta min and max in degrees this time !
    import matplotlib.pyplot as plt
    r = np.zeros(n_points, dtype=complex)
    t = np.zeros(n_points, dtype=complex)
    R = np.zeros(n_points)
    T = np.zeros(n_points)
    incidence = np.zeros(n_points)
    incidence = np.linspace(theta_min, theta_max, n_points)
    for k in range(n_points):
        r[k], t[k], R[k], T[k] = coefficient(structure, wavelength,
                                             incidence[k] / 180 * np.pi,
                                             polarization)

    return incidence, r, t, R, T


# For inspiration



# TODO: Probably also add spectrum_absorption and angular_absorpiont
# (make it an option of spectrum/angular?)

# def absorption_A_opti_wavelength(struct, wavelength, incidence, polarization): #numpy friendly
#     """
#     This function computes the percentage of the incoming energy
#     that is absorbed in each layer when the structure is illuminated
#     by a plane wave.

#     Args:
#         struct (Structure): belongs to the Structure class
#         wavelength (float): wavelength of the incidence light (in nm)
#         incidence (float): incidence angle in radians
#         polarization (float): 0 for TE, 1 (or anything) for TM

#     returns:
#         absorb (numpy array): absorption in each layer
#         r (complex): reflection coefficient, phase origin at first interface
#         t (complex): transmission coefficient
#         R (float): Reflectance (energy reflection)
#         T (float): Transmittance (energie transmission)

#     R and T are the energy coefficients (real quantities)

#     .. warning: The transmission coefficients have a meaning only if the lower medium
#     is lossless, or they have no true meaning.
#     """
#     # In order to get a phase that corresponds to the expected reflected coefficient,
#     # we make the height of the upper (lossless) medium vanish. It changes only the
#     # phase of the reflection coefficient.

#     # The medium may be dispersive. The permittivity and permability of each
#     # layer has to be computed each time.
#     len_wl = wavelength.size
#     len_mat = len(struct.materials)
#     wavelength.shape = (len_wl, 1)

#     if (struct.unit != "nm"):
#         wavelength = conv_to_nm(wavelength, struct.unit)

#     # Epsilon and Mu are (len_wl, len_mat) arrays.
#     Epsilon, Mu = struct.polarizability_opti_wavelength(wavelength)
#     Epsilon.shape, Mu.shape = (len_wl, len_mat), (len_wl, len_mat)
#     thickness = copy.deepcopy(struct.thickness)
#     thickness = np.asarray(thickness)

#     # In order to ensure that the phase reference is at the beginning
#     # of the first layer.
#     thickness[0] = 0
#     thickness.shape = (1, len(thickness))
#     Type = struct.layer_type
#     # The boundary conditions will change when the polarization changes.
#     if polarization == 0:
#         f = Mu
#     else:
#         f = Epsilon

#     # Wavevector in vacuum. Array of shape (len_wl, 1).
#     k0 = 2 * np.pi / wavelength

#     # Number of layers.
#     g = len(struct.layer_type)

#     # Wavevector k_x, horizontal. Array of shape (len_wl, 1).
#     Epsilon_first, Mu_first = Epsilon[:,Type[0]], Mu[:,Type[0]]
#     Epsilon_first.shape, Mu_first.shape = (len_wl, 1), (len_wl, 1)
#     alpha = np.sqrt(Epsilon_first * Mu_first) * k0 * np.sin(incidence)
#     # Computation of the vertical wavevectors k_z. Array of shape (len_wl, len_mat).
#     gamma = np.sqrt(
#         Epsilon[:,Type] * Mu[:,Type] * k0 ** 2 - np.ones((len_wl, g)) * alpha ** 2)

#     # Be cautious if the upper medium is a negative index one.
#     mask = np.logical_and(np.real(Epsilon_first) < 0, np.real(Mu_first) < 0)
#     np.putmask(gamma[:,0], mask,-gamma[:,0])

#     # Changing the determination of the square root to achieve perfect stability.
#     if g > 2:
#         gamma[:,1:g - 2] = gamma[:,1:g - 2] * (
#                     1 - 2 * (np.imag(gamma[:,1:g - 2]) < 0))

#     # Outgoing wave condition for the last medium.
#     Epsilon_last, Mu_last = Epsilon[:,Type[g-1]], Mu[:,Type[g-1]]
#     Epsilon_last.shape, Mu_last.shape = (len_wl, 1), (len_wl, 1)
#     gamma_last = np.sqrt(Epsilon_last* Mu_last * k0 ** 2 - alpha ** 2)
#     mask = np.logical_and.reduce(
#     (np.real(Epsilon_last) < 0, np.real(Mu_last) < 0, np.real(gamma_last) != 0))
#     not_mask = np.logical_or.reduce(
#     (np.real(Epsilon_last) > 0, np.real(Mu_last) > 0, np.real(gamma_last) == 0))
#     np.putmask(gamma[:,g-1], mask, -gamma_last)
#     np.putmask(gamma[:,g-1], not_mask, gamma_last)

#     # Each layer has a (2, 2) matrix with (len_wl, 1) array as coefficient.
#     T = np.zeros(((g-1, 2, 2, len_wl)), dtype=np.clongdouble)
#     c = np.cos(gamma * thickness)
#     s = np.sin(gamma * thickness)
#     gf = gamma/f[:,Type]

#     for k in range(g-1):
#         # Layer scattering matrix
#         ephemeral_c_k, ephemeral_s_k, ephemeral_gf_k = c[:,k], s[:,k], gf[:,k]
#         ephemeral_c_k.shape, ephemeral_s_k.shape, ephemeral_gf_k.shape = (len_wl), (len_wl), (len_wl)
#         T[k] = np.array([[ephemeral_c_k, -ephemeral_s_k / ephemeral_gf_k],
#                   [ephemeral_gf_k * ephemeral_s_k, ephemeral_c_k]])

#     # Once the scattering matrixes have been prepared, now let us combine them

#     A = np.empty((T.shape[0], 2, 2, len_wl), dtype=np.clongdouble)
#     A[0] = T[0]

#     # We change the form of the matrix A to use numpy methods.
#     for i in range(1, T.shape[0]):
#         Y = np.transpose(A[i-1], (2,0,1))
#         X = np.transpose(T[i], (2,0,1))
#         Z = np.matmul(X, Y)
#         A[i] = np.transpose(Z, (1,2,0))

#     a = A[-1][0, 0][:]
#     b = A[-1][0, 1][:]
#     c = A[-1][1, 0][:]
#     d = A[-1][1, 1][:]

#     amb = a - 1.j * gf[:,0] * b
#     apb = a + 1.j * gf[:,0] * b
#     cmd = c - 1.j * gf[:,0] * d
#     cpd = c + 1.j * gf[:,0] * d

#     # reflection coefficient of the whole structure
#     r = -(cmd + 1.j * gf[:,-1] * amb)/(cpd + 1.j * gf[:,-1] * apb)
#     # transmission coefficient of the whole structure
#     t = a * (r+1) + 1.j * gf[:,0] * b * (r-1)
#     # Energy reflexion coefficient;
#     R = np.real(np.absolute(r) ** 2)
#     # Energy transmission coefficient;
#     T = np.real(np.absolute(t) ** 2 * gf[:,g - 1] / gf[:,0])


#     I = np.zeros(((A.shape[0]+1, 2, len_wl)), dtype=complex)

#     for k in range(A.shape[0]):
#         I[k,0][:] = A[k][0, 0][:] * (r + np.ones_like(r)) + A[k][0, 1][:]*(1.j*gf[:,0]*(r - np.ones_like(r)))
#         I[k,1][:] = A[k][1, 0][:] * (r + np.ones_like(r)) + A[k][1, 1][:]*(1.j*gf[:,0]*(r - np.ones_like(r)))
#         # Contains Ey and dzEy in layer k
#     I[-1,:] = [t, -1.j*gf[:,-1]*t]

#     w = 0
#     poynting = np.zeros((A.shape[0]+1, len_wl), dtype=complex)
#     if polarization == 0:  # TE
#         for k in range(A.shape[0]+1):
#             poynting[k,:] = np.real(-1.j * I[k, 0,:] * np.conj(I[k, 1,:]) / gf[:,0])
#     else:  # TM
#         for k in range(A.shape[0]+1):
#             poynting[k,:] = np.real(1.j * np.conj(I[k, 0,:]) * I[k, 1,:] / gf[:,0])
#     # Absorption in each layer

#     zeros = np.zeros((1, len_wl))
#     diff_poynting = abs(-np.diff(poynting, axis=0))
#     absorb = np.concatenate((zeros, diff_poynting), axis=0)
#     absorb = np.transpose(absorb)
#     # First layer is always supposed non absorbing

#     return absorb, r, t, R, T

