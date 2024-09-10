import numpy as np
from context import PM
import matplotlib.pyplot as plt
materials = []

wavelength = 600

def bragg(x):
  # This cost function corresponds to the problem
  # of maximizing the reflectance, at a given wavelength,
  # of a multilayered structure with alternating refractive
  # indexes. This problem is inspired by the first cases studied in 
  # https://www.nature.com/articles/s41598-020-68719-3
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  # The input, x, corresponds to the thicknesses of all the layers, :
  # starting with the upper one. It should be a python list, not a  :
  # numpy array, with an even number of elements.                   :
  # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
  x = list(x)
  n = len(x)
  # Working wavelength
  wl = 600.
  materials = [1,1.4**2,1.8**2]
  stack = [0] + [2,1] * (n//2) + [2]
  thicknesses = [0.] + x + [0.]
  structure = PM.Structure(materials,stack,thicknesses,verbose = False)
  _,_,R,_ = PM.coefficient(structure,wl,0.,0)
  cost = 1-R

  return cost

nb_layers = 20
min_th = 50
max_th = 200

X_min = np.array([min_th]*nb_layers)
X_max = np.array([max_th]*nb_layers)

budget = 10000

start = np.array([np.random.random()*(max_th-min_th) + min_th for i in range(nb_layers)])
best_b, cost_b = PM.bfgs(bragg, budget, start)

print(cost_b)

best, convergence = PM.differential_evolution(bragg, budget, X_min, X_max, f1=0.9, f2=0.8, cr=0.7)

print(convergence[-1])


best_saqn, convergence = PM.SAQNDE(bragg, budget, X_min, X_max, f1=0.5, f2=[0.6, 0.8], cr=[0.5, 0.6, 0.7])

plt.plot(best_b, label="bfgs solution")
plt.plot(best, label="de solution")
plt.plot(best_saqn, label="SAQNde solution")
plt.legend()
plt.show()
