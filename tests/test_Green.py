import numpy as np
from context import PM
import matplotlib.pyplot as plt

green = PM.Structure([1.,4+0.1j],[0,1,1,0],[1000,500,500,1000])
polarization = 1
wavelength = 800
window = PM.Window(30*wavelength,0.5,5.,5.)
source_interface = 2

En = PM.Green(green,window,wavelength,source_interface)
plt.figure(2)
plt.imshow(abs(En),cmap='jet',aspect='auto')
plt.colorbar()
plt.show()
