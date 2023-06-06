import numpy as np
from context import PM
import matplotlib.pyplot as plt

green = PM.Structure([1,4+0.1j],[0,1,1,0],[2000,500,500,2000])
wavelength = 800
window = PM.Window(30*wavelength,0.5,10.,10.)
# Just to make sure...
print(PM.__version__)

source_interface = 2
En = PM.green(green,window,wavelength,source_interface)
#plt.imshow(abs(np.real(En)),cmap='jet',aspect='auto')
#plt.colorbar()

plt.imsave('champ.png',abs(np.real(En)),cmap = 'jet')
