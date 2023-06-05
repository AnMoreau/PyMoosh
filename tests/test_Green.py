import numpy as np
from context import PM
import matplotlib.pyplot as plt

<<<<<<< HEAD
green = PM.Structure([1.,4+0.1j],[0,1,1,0],[2000,500,500,2000])
polarization = 1
wavelength = 800
window = PM.Window(30*wavelength,0.5,3.,3.)
=======
green = PM.Structure([1.,4+0.1j],[0,1,1,0],[1000,500,500,1000])
polarization = 1
wavelength = 800
window = PM.Window(30*wavelength,0.5,5.,5.)
>>>>>>> 3239dc9fcc679fc61d0bb7a82b59b3e67d2c509a
source_interface = 2

En = PM.Green(green,window,wavelength,source_interface)
plt.figure(2)
<<<<<<< HEAD
plt.imshow(abs(np.real(En)),cmap='jet',aspect='auto')
plt.colorbar()
=======
plt.imshow(abs(En),cmap='jet',aspect='auto')
plt.colorbar()
plt.show()
>>>>>>> 3239dc9fcc679fc61d0bb7a82b59b3e67d2c509a
