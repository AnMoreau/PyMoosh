import numpy as np
from context import PM
from context import green
import matplotlib.pyplot as plt

green_struct = PM.Structure([1, 4 + 0.1j], [0, 1, 1, 0], [2000, 500, 500, 2000])
wavelength = 800
window = PM.Window(30 * wavelength, 0.5, 10.0, 10.0)


source_interface = 2
En = green.green(green_struct, window, wavelength, source_interface)
# plt.imshow(abs(np.real(En)),cmap='jet',aspect='auto')
# plt.colorbar()

plt.imsave("champ.png", abs(np.real(En)), cmap="jet")
