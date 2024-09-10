import numpy as np
import PyMoosh as PM
import matplotlib.pyplot as plt
import csv
from PyMoosh.non_local import *

# Lets read the external experimental data and the wavelength list.
# This will be useful when we call the cost function.
wavelength_list = np.linspace(5000, 8000, 3001)

with open('nlplot.data', newline='') as csvfile:
    expdata = list(csv.reader(csvfile))[0]
expdata = [float(p.strip()) for p in expdata]
expdata = np.array(expdata)
refplot, = plt.plot(wavelength_list, expdata, label="Reference")
#plt.show()

def dopedsc_basic(wavelength):
    """
    Function returning the characteristics of the response of a
    doped semiconductor in the infra-red range. It is a non dispersive
    background permittivity with a Drude part. For nonlocality to be
    taken into account beta**2 is required, as well as a distinction between
    the permittivity of the background (chi_b) and the effective one of the
    electron gas (chi_f). The plasma frequency w_p is also required.

    Here beta**2 is complex, but does not change with the frequency.
    """
    chi_b = 11.6644913
    w_p = 8.9e+14
    gamma = 3.53201120e+12
    beta2 = 1.92864068e+30
    w = 2*np.pi*299792458*1e9 / wavelength
    # We put a negative imaginary part for the losses that are intrinsic
    # to the electron gas. Nothing dispersive here.
    eta2 = beta2 - 1.0j*2.2881657413884294e+29
    chi_f = - w_p**2/(w * (w + 1j * gamma))
    return eta2, chi_b, chi_f, w_p


# Now that we have defined the functions, we can instanciate (declare)
# and create a material nSC (n-doped semiconductor) that is nonlocal.
# You juste have to provide a function that provides the parameters.
nSC = NLMaterial(dopedsc_basic)
# Now we define the materials we use
materials = [15.6816, nSC, 14.2129]
# And how we stack them.
stack = [0, 1, 2]
# Their thickness. For the superstrate and the substrate, thickness makes
# no sense so let's put it to 0.
thickness = [0, 105, 0]
theta = np.pi * 37 / 180
pol = 1.0

simple_nl = NLStructure(materials,stack,thickness)
R_modelsimple = []
for wl in wavelength_list:
    _,_,R,_ = NLcoefficient(simple_nl,wl,theta,pol)
    R_modelsimple.append(R)
R=np.array(R_modelsimple)
R = R/max(R)*max(expdata)
simpleplot, = plt.plot(wavelength_list,R,label="Simple NL")

def sc_2ndviscosity(wavelength, chi_b, w_p, gamma, beta_0, tau):
    """
    Function returning the characteristics of the response of a
    doped semiconductor in the infra-red range. It is a non dispersive
    background permittivity with a Drude part. For nonlocality to be
    taken into account beta**2 is required, as well as a distinction between
    the permittivity of the background (chi_b) and the effective one of the
    electron gas (chi_f). The plasma frequency w_p is also required.

    Here beta**2 is complex and the imaginary part is dispersive, because of
    the _second viscosity_ in electron gases.

    The parameters of the model have to be provided when declaring
    the material as nonlocal.
    """
    w = 2*np.pi*299792458*1e9 / wavelength
    eta2 = beta_0**2 - 1.0j * w * tau
    chi_f = - w_p**2/(w * (w + 1j * gamma))

    return eta2, chi_b, chi_f, w_p

chi_b = 11.6644913
w_p = 8.9e+14
gamma = 3.53201120e+12
beta2 = 1.92864068e+30
nSC_2nd = NLMaterial([sc_2ndviscosity,chi_b,w_p,gamma,np.sqrt(beta2),9.718e14])

advanced_nl = NLStructure([15.6816, nSC_2nd, 14.2129],stack,thickness)
R_advanced = []
for wl in wavelength_list:
    _,_,R,_ = NLcoefficient(advanced_nl,wl,theta,pol)
    R_advanced.append(R)
R=np.array(R_advanced)
R = R/max(R)*max(expdata)
advancedplot, = plt.plot(wavelength_list,R,label="Advanced NL")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectivity")

plt.legend()
plt.show()


plt.clf()

refplot, = plt.plot(wavelength_list, expdata, label="Reference")

def cost_function(X):
    chi_b, w_p, gamma, beta_0, tau, base, scale = X
    sc_nl = NLMaterial([sc_2ndviscosity, chi_b, w_p, gamma, beta_0, tau])
    nb_lam = 50
    materials = [15.6816, sc_nl, 14.2129]
    stack = [0, 1, 2]
    thickness = [0, 105, 0]
    thing = NLStructure(materials,stack,thickness,verbose=False)
    new_wl = np.linspace(5000,8000,nb_lam)
    R=np.zeros(nb_lam)
    for k in range(nb_lam):
        _,_,R[k],_ = NLcoefficient(thing,new_wl[k],np.pi * 37 / 180,1.0)
    R = R*scale + base
    obj  = np.interp(new_wl, wavelength_list, expdata)
    cost = np.mean(np.abs(R - obj))+nb_lam*np.mean(np.abs(np.diff(R)-np.diff(obj)))
    return cost/nb_lam

X_min = np.array([11.4, 1e14, 1e12, 1e14, 6e14, 0, 0.01])
X_max = np.array([11.8, 1e15, 1e13, 1.5e15, 1.2e15, 0.2, 2])

best , convergence = PM.QODE(cost_function,2000,X_min,X_max,progression=10)
chi_b, w_p, gamma, beta_0, tau, base, scale = best

nSC_2nd = NLMaterial([sc_2ndviscosity,chi_b, w_p, gamma, beta_0, tau])
optimized_nl = NLStructure([15.6816, nSC_2nd, 14.2129],stack,thickness)
R_optim = []
for wl in wavelength_list:
    _,_,R,_ = NLcoefficient(optimized_nl,wl,theta,pol)
    R_optim.append(R)
R=np.array(R_optim)
R = R*scale+base
optimizedplot, = plt.plot(wavelength_list,R,label="Optimized NL")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectivity")

plt.legend()
plt.show()
