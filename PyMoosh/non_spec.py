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
    E = field(struct, beam, window, True)
    layer = struct.thickness
    Pixel_layer = abs(int(layer[0]/window.py) - 1)
    if type == "abs":
        inter_up = abs(E[Pixel_layer])
    else:
        inter_up = E[Pixel_layer]
    return inter_up

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
    E = field(struct, beam, window, False)
    layer = struct.thickness
    Pixel_layer = abs(int(layer[0]/window.py) - 1)
    #-1 because the intern reflexions at the interface counts as incoming beam, we take the "upper" interface to keep only the true incoming beam
    if type == "abs":
        inter_up = abs(E[Pixel_layer])
    else:
        inter_up = E[Pixel_layer]
    return inter_up

#Profil of the wave amplitude along the first surface layer (in all directions)
def profil(struct, beam, window, type = "abs"):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")
        ty (str, optional): "abs" by default, return the amplitude field. If changed the entire field will be returned (complex value)

    Returns:
        Field (Array): return the field along the first interface
    """
    E = field(struct, beam, window)
    layer = struct.thickness
    Pixel_layer = abs(int(layer[0]/window.py) - 1)
    if type == "abs":
        inter_up = abs(E[Pixel_layer])
    else:
        inter_up = E[Pixel_layer]
    return inter_up

def deltas(struct, beam, window):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used
        window (Window): the window used (big enough to avoid resonances during computation using the function "field")

    Returns:
        delta (float): the shift between the center of the incident and the reflected beam
        Delta (float): the enlargement of the reflected beam compared to the incident beam
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
    
    return float(delta), float(Delta)

def asymptcoef(struct, beam):
    """
    Args:
        struct (Structure): the structure used
        beam (Beam): the beam used

    Returns:
        delta (float):
        Delta (float): the enlargement of the reflected beam compared to the incident beam
    """
    lam = beam.wavelength
    pol = beam.polarization
    theta = beam.incidence
    nk0 = np.sqrt(struct.materials[0].permittivity) * 2 * np.pi / lam
    dtheta = np.pi / (180 * 100)
    phase = []    
    R = []
    N = 101
    Theta = np.linspace(-dtheta, dtheta, N)
    for i in  Theta:
        r = coefficient_S(struct, lam, (theta + i), pol)[0]
        phase.append(np.angle(r))
        R.append(np.abs(r))
        # kx = n*ko*sin(theta) so d/d(kx) = d/(alpha * d(theta))  = 1/alpha * d/d(theta)
    c0 = nk0 * np.cos(theta)
    phase = np.array(phase)
    # We have to supressed the 2*pi jumped in the angular list (especially around limit angle) so by using unwrap we will get a better continuity (useful for the gradient)
    phase = np.unwrap(phase) 
    # Because of the size of the array we must redefine dtheta
    dtheta *= 2 / (N-1)
    dphase = np.gradient((phase), dtheta)/(c0)
    R = np.array(R)
    dR = np.gradient(R, dtheta)
    dc0 = nk0 * np.sin(theta) / c0**3
    # Reshaping the previous arrays to avoid incertitude propagation (the first and last values of a gradient array are wrong)
    ddR = np.gradient(dR, dtheta)
    # Now we take the centred values (corresponding to the right incidence angle)
    R = R[int(N/2)]
    dR = dR[int(N/2)]
    ddR = ddR[int(N/2)]
    # d**2/d(kx)**2 = d/d(kx) * d/d(theta) * 1/c0 = 1/c0**2 * d**2/d(theta)**2 + 1/c0 * d(1/c0)/d(theta) * d/d(theta) --- d(1/c0)/d(theta) = nk0*sin(theta)/c0**3 = tan(theta)/c0**2 = dc0
    dR2 = dc0 * dR + ddR / c0**2
    widthlim = (dR**2 / (c0 * R)**2 - dR2 / R) / 2
    return -float(np.real(dphase[int(N/2)])), float(np.real(widthlim))
