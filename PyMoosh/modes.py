"""
This file contain all functions linked to mode finding and plotting
"""

import numpy as np
from PyMoosh.classes import conv_to_nm
from PyMoosh.core import cascade
import copy
import itertools
import matplotlib.pyplot as plt


def dispersion(alpha, struct, wavelength, polarization):
    """It would probably be better to compute the dispersion relation of a
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
        struct (Structure) : the object describing the multilayer
        wavelength : the wavelength in vacuum in nanometer
        polarization : 0 for TE, 1 for TM.

    Returns:
        1/abs(r) : inverse of the modulus of the reflection coefficient.

    """

    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)
    Epsilon, Mu = struct.polarizability(wavelength)
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer. Totally necessary when you are looking for
    # modes of the structure, this makes the poles of the reflection
    # coefficient much more visible.
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
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k0**2 - np.ones(g) * alpha**2)

    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))
    # Changing the determination of the square root in the external medium
    # to better see the structure of the complex plane.
    gamma[0] = gamma[0] * (1 - 2 * (np.angle(gamma[0]) < -np.pi / 5))
    gamma[g - 1] = gamma[g - 1] * (1 - 2 * (np.angle(gamma[g - 1]) < -np.pi / 5))

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

    return 1 / np.abs(r)


#    return 1/r


def complex_map(
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
            T[k, l] = 1 / dispersion(M[k, l], struct, wavelength, polarization)

    return X, Y, T


def guided_modes(
    struct, wavelength, polarization, neff_min, neff_max, initial_points=40
):
    """This function explores the complex plane, looking for zeros of the
    dispersion relation. It does so by launching a steepest descent for a number
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
        #        solution = optim.newton(dispersion,kx,args=(struct,wavelength,polarization),tol=tolerance,full_output = True)
        #        solution = optim.minimize(dispersion,kx,args=(struct,wavelength,polarization))
        solution = steepest(neff, tolerance, 1000, struct, wavelength, polarization)
        #        print(solution)
        if len(modes) == 0:
            modes.append(solution)
        elif min(abs(modes - solution)) > 1e-5 * k_0:
            modes.append(solution)

    return modes


def follow_guided_modes(
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
    dispersion relation. It does so by launching a steepest descent for a number
    `initial_points` of points on the real axis between neff_min and neff_max.


    Args:
        struct (Structure): object describing the multilayer
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
        solution = steepest(neff, tolerance, 1000, struct, wavelength, polarization)
        #        print(solution)
        if len(first_modes) == 0:
            first_modes.append(solution)
        elif min(abs(first_modes - solution)) > 1e-5:
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
                solution = steepest(
                    neff, tolerance, 1000, struct, wavelength, polarization
                )
                if len(new_modes) == 0:
                    new_modes.append(solution)
                elif min(abs(new_modes - solution)) > 1e-5:
                    new_modes.append(solution)
        else:
            neff = neff_start
            solution = steepest(neff, tolerance, 1000, struct, wavelength, polarization)
            if len(new_modes) == 0:
                new_modes.append(solution)
            elif min(abs(new_modes - solution)) > 1e-5:
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


def steepest(start, tol, step_max, struct, wl, pol):
    """Steepest descent to find a zero of the `dispersion`
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
    current = dispersion(z, struct, wl, pol)

    while (current > tol) and (step < step_max):

        grad = (
            dispersion(z + dz, struct, wl, pol)
            - current
            #        -dispersion(z-dz,struct,wl,pol)
            + 1j
            * (
                dispersion(z + 1j * dz, struct, wl, pol)
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

        value_new = dispersion(z_new, struct, wl, pol)
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
        print("Warning: maximum number of steps reached. Final n_eff:", z / k_0)

    return z / k_0


def profile(struct, n_eff, wavelength, polarization, pixel_size=3):

    if struct.unit != "nm":
        wavelength = conv_to_nm(wavelength, struct.unit)
    # Wavevector in vacuum.
    k_0 = 2 * np.pi / wavelength
    # Wavevector of the mode considered here.
    alpha = n_eff * k_0
    # About the structure:
    Epsilon, Mu = struct.polarizability(wavelength)
    thickness = copy.deepcopy(struct.thickness)
    Type = struct.layer_type
    g = len(Type) - 1
    # The boundary conditions will change when the polarization changes.
    if polarization == 0:
        f = Mu
    else:
        f = Epsilon
    # Computation of the vertical wavevectors k_z
    gamma = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2 - np.ones(g + 1) * alpha**2)
    # Changing the determination of the square root to achieve perfect stability
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))
    # Don't forget the square root has to change
    # when the wavevector is complex (same as with
    # dispersion and Map)
    gamma[0] = gamma[0] * (1 - 2 * (np.angle(gamma[0]) < -np.pi / 5))
    gamma[g] = gamma[g] * (1 - 2 * (np.angle(gamma[g]) < -np.pi / 5))
    # We compute all the scattering matrixes starting with the second layer
    T = np.zeros((2 * g, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    gf = gamma / f[Type]
    for k in range(1, g):
        t = np.exp(1j * gamma[k] * thickness[k])
        T[2 * k - 1] = np.array([[0, t], [t, 0]])
        b1 = gf[k]
        b2 = gf[k + 1]
        T[2 * k] = np.array([[b1 - b2, 2 * b2], [2 * b1, b2 - b1]]) / (b1 + b2)
    t = np.exp(1j * gamma[g] * thickness[g])
    T[2 * g - 1] = np.array([[0, t], [t, 0]])

    H = np.zeros((len(T) - 1, 2, 2), dtype=complex)
    A = np.zeros((len(T) - 1, 2, 2), dtype=complex)
    H[0] = T[2 * g - 1]
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

    # Coefficients, in each layer
    Coeffs = np.zeros((g + 1, 2), dtype=complex)
    Coeffs[0] = np.array([0, 1.0])
    # Value of the first down propagating plane wave below
    # the first interface, entering the scattering matrix
    # for the rest of the structure. The amplitude of the
    # incident wave is thus not 1.
    b1 = gamma[0] / f[Type[0]]
    b2 = gamma[1] / f[Type[1]]
    tmp = (b2 - b1) / (2 * b2)
    for k in range(g):
        Coeffs[k + 1] = tmp * np.array([I[2 * k][0, 0], I[2 * k + 1][1, 0]])

    n_pixels = np.floor(np.array(thickness) / pixel_size)
    n_pixels.astype(int)
    n_total = int(np.sum(n_pixels))
    E = np.zeros(n_total, dtype=complex)
    h = 0.0
    t = 0

    for k in range(g + 1):
        for m in range(int(n_pixels[k])):
            h = h + pixel_size
            E[t] = Coeffs[k, 0] * np.exp(1j * gamma[k] * h) + Coeffs[k, 1] * np.exp(
                1j * gamma[k] * (thickness[k] - h)
            )
            t += 1
        h = 0

    x = np.linspace(0, sum(thickness), len(E))
    return x, E
