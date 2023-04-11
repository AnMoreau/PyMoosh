import numpy as np
from context import PM
import matplotlib.pyplot as plt
materials = []

wavelength = 600
kr = PM.Structure([1.,'Gold','Water',1.46**2,1.7**2],[4,1,0],[500,40,500])
polarization = 1
incidence,r,t,R,T=PM.Angular(kr,wavelength,polarization,0.,80.,400)
plt.plot(incidence,R)
plt.show()

window = PM.Window(70*wavelength,0.4,5.,5.)
beam = PM.Beam(wavelength,38.7/180*np.pi,polarization,10*wavelength)
E,Hx,Hz=PM.fields(kr,beam,window)
plt.figure(2)
plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(kr.thickness)],aspect='auto')
plt.colorbar()
plt.show()


plt.imshow(abs(Hx),cmap='jet',extent=[0,window.width,0,sum(kr.thickness)],aspect='auto')
plt.colorbar()
plt.show()

plt.imshow(abs(Hz),cmap='jet',extent=[0,window.width,0,sum(kr.thickness)],aspect='auto')
plt.colorbar()
plt.show()
