import numpy as np
from context import PM
import matplotlib.pyplot as plt
import itertools


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [1., 2.+0.1j]
stack = [0,1,0]
thickness = [0,500,0]
formap = PM.Structure(materials, stack, thickness)

X,Y,T = PM.complex_map(formap,600.,0.,[1.,np.sqrt(2)],[-0.1,0.1],100,100)

plt.imshow(np.log(np.abs(T)), cmap='jet', extent=[X.min(), X.max(), Y.min(), Y.max()])
#plt.imshow(np.angle(T), cmap='jet', extent=[X.min(), X.max(), Y.min(), Y.max()])

#plt.colorbar()
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()
