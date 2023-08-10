import PyMoosh as pm
import numpy as np
import matplotlib.pyplot as plt


def test_pymoosh():
  def bragg(x):
    # This cost function corresponds to the problem
    # of maximizing the reflectance, at a given wavelength,
    # of a multilayered structure with alternating refractive
    # indexes. This problem is inspired by the first cases studied in
    # https://www.nature.com/articles/s41598-020-68719-3
    # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
    # The input, x, corresponds to the thicknesses of all the layers, :
    # starting with the upper one.                                    :
    # :-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
    x = list(x)
    n = len(x)
    # Working wavelength
    wl = 600.
    materials = [1,1.4**2,1.8**2]
    stack = [0] + [2,1] * (n//2) + [2]
    thicknesses = [0.] + list(x) + [0.]
    structure = pm.Structure(materials,stack,np.array(thicknesses),verbose = False)
    _, R = pm.coefficient_I(structure,wl,0.,0)
    cost = 1-R
    return cost
  
  budget = 200
  nb_layers = 12
  dim = nb_layers
  opti_wave = 600
  mat1 = 1.4
  mat2 = 1.8
  min_th = 0 # We don't want negative thicknesses.
  max_th = opti_wave/(2*mat1) # A thickness of lambda/2n + t has the same behaviour as a thickness t

  X_min = np.array([min_th]*nb_layers)
  X_max = np.array([max_th]*nb_layers)


  best, convergence = pm.differential_evolution(bragg, budget, X_min, X_max)

  plt.plot(convergence)
  plt.xlabel("Optimization step")
  plt.ylabel("Cost")
  plt.show()

  # Showing the final spectrum
  materials = [1,mat1**2,mat2**2]
  stack = [0] + [2,1] * (nb_layers//2) + [2]
  thicknesses = [0.] + [t for t in best] + [0.]

  bragg_mirror = [opti_wave / (4*np.sqrt(materials[2])), opti_wave / (4*np.sqrt(materials[1]))] * (nb_layers//2)
  bragg_th = [0.] + [t for t in bragg_mirror] + [0.]
  structure = pm.Structure(materials,stack,thicknesses,verbose = False)
  bragg_structure = pm.Structure(materials,stack,bragg_th,verbose = False)

  wavelengths = np.linspace(opti_wave-150, opti_wave+150, 300)
  spectrum = np.zeros_like(wavelengths)
  bragg_spectrum = np.zeros_like(wavelengths)
  for i, wav in enumerate(wavelengths):
    _,_,R,_ = pm.coefficient(structure,wav,0.,0)
    spectrum[i] = R
    _,_,R,_ = pm.coefficient(bragg_structure,wav,0.,0)
    bragg_spectrum[i] = R

 # plt.plot(wavelengths, spectrum, label="Optimized structure")
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_cv.png")
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_cv.svg")
  plt.clf()
  plt.plot(wavelengths, bragg_spectrum, label="Bragg mirror")
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Reflectivity")
  plt.legend()
  plt.show()

  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg.png")
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg.svg")
  plt.clf()
  structure.plot_stack(wavelength=opti_wave)
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_stack.png")
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_stack.svg")
  plt.clf()
  bragg_structure.plot_stack(wavelength=opti_wave)
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_structure.png")
  plt.savefig(f"phy_dim{dim}_budget{budget}_bragg_structure.svg")

