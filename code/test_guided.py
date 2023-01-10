from PyMoosh import *
import matplotlib.pyplot as plt

materials = [1.,2.]
stack = [0,1,0,1,0]
thickness = [200,1300,300,1300,200]
guide = Structure(materials,stack,thickness)

wavelength = 600
polarization = 0.
modes = Guided_modes(guide,wavelength,0.,1.001,1.45)

plt.figure(1)
for k in range(len(modes)):
    x,prof = Profile(guide,modes[k],wavelength,polarization)
    plt.plot(x,np.real(prof),linewidth = 2)
#plt.ylabel('Mode profile (a.u), intensity')
plt.show()
