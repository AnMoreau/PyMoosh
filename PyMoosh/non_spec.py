import numpy as np
import matplotlib.pyplot as plt
from PyMoosh.classes import *
from PyMoosh.core import *


#Profil of the reflexion wave amplitude along the first surface layer
def profil_R(struct, beam, window, type = "abs"):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")
        type (str, optional): "abs" by default, return the amplitude field. If changed the entire field will be returned (complex value)

    Returns:
        Field (Array): return the reflected field along the first interface
    """
    E = NSfield(struct, beam, window, True)
    if type == "abs":
        inter = abs(E)
    else:
        inter = E
    return inter

#Profil of the incoming wave amplitude along the first surface layer
def profil_In(struct, beam, window, type = "abs"):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")
        ty (str, optional): "abs" by default, return the amplitude field. If changed the entire field will be returned (complex value)

    Returns:
        Field (Array): return the incident field along the first interface
    """
    E = NSfield(struct, beam, window, False)
    if type == "abs":
        inter = abs(E)
    else:
        inter = E
    return inter

#Profil of the wave amplitude along the first surface layer (in all directions)
def profil_tot(struct, beam, window, type = "abs"):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")
        ty (str, optional): "abs" by default, return the amplitude field. If changed the entire field will be returned (complex value)

    Returns:
        Field (Array): return the field along the first interface
    """
    E = NSfield(struct, beam, window)
    if type == "abs":
        inter = abs(E)
    else:
        inter = E
    return inter

def deltas(struct, beam, window, rel = True):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")
        rel (Boolean): "True" by default. If rel is "True" the function will return relative shift and enlargement. If not it will return the algebric values
    Returns:
        delta (float): the shift of the reflected beam compared to the incident beam (relative if rel is "True", algebric otherwise)
        Delta (float): the enlargement of the reflected beam compared to the incident beam (relative if rel is "True", algebric otherwise)
    """
    midIn = 0
    midOut = 0
    enter = profil_In(struct, beam, window)
    outer = profil_R(struct, beam, window)
    X = np.arange(0,len(enter)) * window.px # * window.px to get the physical shift and not the pixel shift
    # The shift:
    midIn = np.trapezoid(X * np.abs(enter)**2)/np.trapezoid(np.abs(enter)**2)
    midOut = np.trapezoid(X * np.abs(outer)**2)/np.trapezoid(np.abs(outer)**2)
    delta = midOut - midIn
    
    # The enlargement:
    varIn = np.trapezoid((X - np.ones(len(X)) * midIn)**2 * np.abs(enter)**2)/np.trapezoid(np.abs(enter)**2)  # We need to shift the center of the variance (the center of the incident beam is not 0)
    varOut = np.trapezoid((X - np.ones(len(X)) * (midIn + delta))**2 * np.abs(outer)**2)/np.trapezoid(np.abs(outer)**2)
    Delta = varOut - varIn
    
    if rel:
        return float(delta) / beam.waist, varOut / beam.waist
    else:
        return float(delta), float(Delta)

def asymptcoef(struct, beam):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used

    Returns:
        delta (float): the shift of the reflected beam compared to the incident beam
        Delta (float): the enlargement of the reflected beam compared to the incident beam
    """
    lam = beam.wavelength
    pol = beam.polarization
    theta = beam.incidence
    nk0 = np.sqrt(struct.materials[0].permittivity) * 2 * np.pi / lam
    dtheta = np.pi / (180 * 10000)
    phase = []    
    R = []
    N = 1001
    Theta = np.linspace(-dtheta, dtheta, N)
    for i in  Theta:
        r = coefficient_S(struct, lam, (theta + i), pol)[0]
        phase.append(np.angle(r))
        R.append(np.abs(r))
        # kx = n*ko*sin(theta) so d/d(kx) = d/(alpha * d(theta))  = 1/alpha * d/d(theta)
    c0 = nk0 * np.cos(theta)
    phase = np.array(phase)
    # Because of the size of the array we must redefine dtheta
    dtheta *= 2 / (N-1)
    dphase = diff((phase), dtheta)/(c0)
    R = np.array(R)
    dR = diff(R, dtheta)
    dc0 = nk0 * np.sin(theta) / c0**3
    # Reshaping the previous arrays to avoid incertitude propagation (the first and last values of a gradient array are wrong)
    ddR = diff(dR, dtheta)
    # Now we take the centred values (corresponding to the right incidence angle)
    R = R[int(N/2)]
    dR = dR[int(N/2)]
    ddR = ddR[int(N/2)]
    # d**2/d(kx)**2 = d/d(kx) * d/d(theta) * 1/c0 = 1/c0**2 * d**2/d(theta)**2 + 1/c0 * d(1/c0)/d(theta) * d/d(theta) --- d(1/c0)/d(theta) = nk0*sin(theta)/c0**3 = tan(theta)/c0**2 = dc0
    dR2 = dc0 * dR + ddR / c0**2
    widthlim = (dR**2 / (c0 * R)**2 - dR2 / R) / 2
    return -float(np.real(dphase[int(N/2)])), float(np.real(widthlim))

def diff(A, step):
    D = []
    for i in range(1, len(A)-2):
        D.append((A[i + 1] - A[i - 1]) / (2 * step))
    return np.array(D)

def NSfield(struct, beam, window, onlyreflected = None):
    """Computes the electric (TE polarization) or magnetic (TM) field inside
    a multilayered structure illuminated by a gaussian beam.

    Args:
        struct (Structure): description (materials,thicknesses)of the multilayer
        beam (Beam): description of the incidence beam
        window (Window): description of the simulation domain
        onlyreflected (None type): None by default, if True it will only shows the reflected field, if False it will only shows the incident field

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
    # Position where the beam waste is taken
    Dphi = beam.distance
    w = beam.waist
    pol = beam.polarization
    d = window.width
    theta = beam.incidence
    pixDphi = Dphi/d
    C = window.C
    ny = np.floor(thickness / window.py)
    nx = window.nx
    Type = struct.layer_type
    transmission = 1
    reflexion = 1
    if onlyreflected:
        transmission = 0
    elif onlyreflected == False:
        reflexion = 0
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
    k_0 = 2 * pi / l
    # Initialization of the field component
    En = []
    # Total number of layers
    # g=Type.size-1
    g = len(struct.layer_type) - 1
    # Amplitude of the different modes
    nmodvect = np.arange(-nmod, nmod + 1)
    # First factor makes the gaussian beam, the second one the shift
    # a constant phase is missing, it's just a change in the time origin.
    gauss = np.exp(-(w**2) * pi**2 * nmodvect**2)
    phase = np.exp(-2 * 1j * pi * nmodvect * C)
    X = gauss * phase 
    
    layer_k = np.sqrt(Epsilon[Type] * Mu[Type] * k_0**2)

    # Scattering matrix corresponding to no interface.
    T = np.zeros((2 * g + 2, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for nm in np.arange(2 * nmod + 1):

        n_0 = np.sqrt(Epsilon[Type[0]] * Mu[Type[0]])
        alpha = n_0 * k_0 * sin(theta) + 2 * pi * (nm - nmod)
        gamma = np.sqrt(layer_k**2 - np.ones(g + 1) * alpha**2)

        if np.real(Epsilon[Type[0]]) < 0 and np.real(Mu[Type[0]]) < 0:
            gamma[0] = -gamma[0]

        if g > 2:
            im_sign = np.imag(gamma[1 : g - 1]) < 0
            gamma[1 : g - 1] = gamma[1 : g - 1] * (1 - 2 * im_sign)
        if (
            np.real(Epsilon[Type[g]]) < 0
            and np.real(Mu[Type[g]]) < 0
            and np.real(np.sqrt(layer_k[g] ** 2 - alpha**2)) != 0
        ):
            gamma[g] = -np.sqrt(layer_k[g] ** 2 - alpha**2)
        else:
            gamma[g] = np.sqrt(layer_k[g] ** 2 - alpha**2)
        
        # Suppression of vanishing mod when the waist distance is not 0, to avoid amplifications and overwhelming amplitudes 
        if (np.imag(gamma[0]) > 0) and (pixDphi > 0):
            gamma[0] = 0
        
        
        gf = gamma / f[Type]
        for k in range(g):
            t = np.exp(1j * gamma[k] * (thickness[k]))
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

        I = np.zeros((2, 2, 2), dtype=complex)
        for k in range(2):
            I[k] = np.array(
                [
                    [A[k][1, 0], A[k][1, 1] * H[len(T) - k - 2][0, 1]],
                    [A[k][1, 0] * H[len(T) - k - 2][0, 0], H[len(T) - k - 2][0, 1]],
                ]
                / (1 - A[k][1, 1] * H[len(T) - k - 2][0, 0])
            )

        h = 0

        E = 0

        for m in range(int(ny[0])):
            h += float(thickness[0]) / ny[0]
        E = transmission*(I[0][0, 0] * np.exp(1j * gamma[0] * h)) + reflexion*(I[1][
            1, 0] * np.exp(1j * gamma[0] * (thickness[0] - h)))
        E = E * np.exp(1j * alpha * np.arange(0, nx) / nx) 
        # We can now add the phase on gamma for the waist distance (we use gamma[0] because the phase is implemented in the first layer)
        E *= np.exp(-1j * gamma[0] * pixDphi)
        En.append(X[int(nm)] * E) 

    return sum(En)
