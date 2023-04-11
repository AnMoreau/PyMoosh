from context import PM
import numpy as np
import matplotlib.pyplot as plt


wavelength=600
#
# TIR=PM.Structure([2.25,1.],[0,1],[20*wavelength,3*wavelength])
#
# # For TE polarization
# incidence,r,t,R,T=PM.Angular(TIR,wavelength,0.,0.,89.,200)
# # For TM polarization, same incidence angles
# incidence,r_p,t_p,R_p,T_p=PM.Angular(TIR,wavelength,1.,0.,89.,200)
#
# # Visualization of the result
# import matplotlib.pyplot as plt
# # plt.figure(1)
# # plt.plot(incidence,R)
# # plt.plot(incidence,R_p)
# # plt.legend(["s","p"])
# #
# # plt.ylabel('Reflectance')
# # plt.ylim(0,1)
# # plt.show()
# #
# # plt.figure(2)
# # plt.plot(incidence,np.angle(r))
# # plt.plot(incidence,np.angle(r_p))
# # plt.legend(["s","p"])
#
# window=PM.Window(70*wavelength,0.2,30.,30.)
# beam=PM.Beam(wavelength,45/180*np.pi,1,10*wavelength)
#
# E=PM.field(TIR,beam,window)
#
# plt.figure(2)
# plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(TIR.thickness)],aspect='auto')
# plt.colorbar()
#
# TIR2=PM.Structure([2.25,1.],[0,1],[4*wavelength,3*wavelength])
# window2=PM.Window(30*wavelength,0.32,10,10)
# E=PM.field(TIR2,beam,window2)
# plt.figure(3)
# plt.imshow(abs(np.abs(E)),cmap='jet',extent=[0,window.width,0,sum(TIR.thickness)])
# plt.figure(4)
# plt.imshow(abs(np.abs(E)),cmap='jet',extent=[0,window.width,0,sum(TIR.thickness)],aspect='auto')
# plt.show()



mat = 1.5
ep = wavelength/(4*mat)
n = 20

Bragg=PM.Structure([1., mat**2],[0,1] + [0, 1]*n + [0],[10*wavelength, ep] + [wavelength/4, ep]*n + [3*wavelength])


# Visualization of the result
import matplotlib.pyplot as plt

window=PM.Window(70*wavelength,0.2,30.,30.)
window.C = 0.5
beam=PM.Beam(wavelength,45/180*np.pi,1,10*wavelength)

E=PM.field(Bragg,beam,window)

plt.figure(2)
plt.imshow(abs(E),cmap='jet',extent=[0,window.width,0,sum(Bragg.thickness)],aspect='auto')
plt.colorbar()
plt.show()
