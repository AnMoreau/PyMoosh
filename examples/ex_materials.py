# encoding utf-8
from materials import MaterialFactory, MaterialEnum
import matplotlib.pyplot as plt
import numpy as np


def ex_material_permittivity():
    # Define wavelength support
    wavelength_list = np.linspace(380, 700, 25)
    # Instantiate figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # List of existing materials, CUSTOM is excluded
    materials = [mat for mat in MaterialEnum if mat != MaterialEnum.CUSTOM]
    # Loop over all materials
    for mat in materials:
        # Get material object
        material = MaterialFactory(mat)
        # Get permittivity
        epsilon = np.zeros_like(wavelength_list, dtype="complex")
        for idx, wavelength in enumerate(wavelength_list):
            epsilon[idx] = material.get_permittivity(wavelength=wavelength)
        # Plot
        ax[0].plot(wavelength_list, np.real(epsilon), label=f"{material.name}")
        ax[1].plot(wavelength_list, np.imag(epsilon), label=f"{material.name}")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel("Wavelength [nm]")
    ax[1].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Permittivity Real")
    ax[1].set_ylabel("Permittivity Imag")
    return [fig]


if __name__ == '__main__':
    figs = []
    figs += ex_material_permittivity()
    plt.show()
