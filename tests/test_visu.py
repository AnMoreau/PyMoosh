import numpy as np
from context import PM
import matplotlib.pyplot as plt
materials = []

wavelength = 600
kr = PM.Structure([1.5,1.7**2, 'Gold'],[0, 1, 0, 1, 0, 2],[0,40, 20, 30, 30,0], unit="um")
polarization = 1


kr.plot_stack()
