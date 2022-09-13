from PyMoosh import *
import numpy as np
import matplotlib.pyplot as plt
from math import *

def brewster():

    wavelength=600
    interface=Structure([1.,2.25],[0,1],[10*wavelength,10*wavelength])
    window=Window(100*wavelength,0.2,30.,30.)
    beam=Beam(wavelength,np.arctan(1.5),1,10*wavelength)

    n_points=100
    angle=np.linspace(0,89,n_points)
    r_s=np.zeros(n_points,dtype=complex)
    t_s=np.zeros(n_points,dtype=complex)
    R_s=np.zeros(n_points)
    T_s=np.zeros(n_points)
    r_p=np.zeros(n_points,dtype=complex)
    t_p=np.zeros(n_points,dtype=complex)
    R_p=np.zeros(n_points)
    T_p=np.zeros(n_points)

    for k in range(n_points):
        r_s[k],t_s[k],R_s[k],T_s[k]=coefficient(interface,wavelength,angle[k]/180*np.pi,0)
        r_p[k],t_p[k],R_p[k],T_p[k]=coefficient(interface,wavelength,angle[k]/180*np.pi,1)

    fig, graph = plt.subplots(2, 2)
    graph[0, 0].plot(angle,R_s)
    graph[0, 0].set_title('R_s')
    graph[0, 1].plot(angle, R_p, 'tab:orange')
    graph[0, 1].set_title('R_p')
    graph[1, 0].plot(angle,np.angle(r_s), 'tab:green')
    graph[1, 0].set_title('Phase r_s')
    graph[1, 1].plot(angle, np.angle(r_p), 'tab:red')
    graph[1, 1].set_title('Phase r_p')

    E=field(interface,beam,window)
    plt.figure(2)
    plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(interface.thickness)],aspect='auto')
    plt.colorbar()

    beam.polarization=0
    E=field(interface,beam,window)
    plt.figure(3)
    plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(interface.thickness)],aspect='auto')
    plt.colorbar()

    plt.show()

def total_internal_reflection():

    wavelength=600
    interface=Structure([1.,2.25],[1,0],[10*wavelength,2*wavelength])
    window=Window(80*wavelength,0.25,5.,5.)
    beam=Beam(wavelength,45/180*np.pi,1,10*wavelength)
    Angular(interface,wavelength,0.,0.,89.,200)

    E=field(interface,beam,window)
    plt.figure(3)
    plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(interface.thickness)],aspect='auto')
    plt.colorbar()
    plt.figure(4)
    F=np.real(E)
    plt.imshow(F*(F>0),cmap='jet',extent=[0,window.width,0,sum(interface.thickness)],aspect='auto')
    plt.colorbar()
    plt.show()

def anti_reflective_coating():
    wavelength=600
    n=sqrt(1.5)
    AR_Coating=Structure([1.,n,2.25],[0,1,2],[1000.,0.,1000.])
    [wl,r,t,R,T]=Spectrum(AR_Coating,0.,0,350.,800.,200)
    plt.figure(1)
    plt.plot(wl,R)
    plt.ylabel('Reflectance')
    plt.ylim(0,1)
    plt.show()

def surface_plasmon()
    wavelength=700
    KR_coupler=Structure([1.,'glass','Au'],[1,2,0],[10*wavelength,45,])
    [incidence,r,t,R,T]=Angular(KR_coupler,wavelength,1.,20.,75.,200)
    plt.figure(1)
    plt.plot(incidence,R)
    plt.ylabel('Reflectance')
    plt.ylim(0,1)
    plt.show()
