# PyMoosh

## About PyMoosh

PyMoosh is the python version of Moosh, an Octave/Matlab code meant as a swiss knife for the study of multilayered structures from an optical point of view.

Not all the features of Moosh have yet been transferred into PyMoosh, but this work is in progress. Plus, given the really nice feedbacks I had recently on PyMoosh and the plans ahead, you can count on the fact that PyMoosh is high on my priority list.

I've recently discovered the Jupyter Notebooks, and I'll use them extensively to illustrate how Moosh works and how much physics/optics can be made with such a tool. I have really discovered over the years how far this may go, and I am pretty sure you will be surprised too. This is the kind of codes we use to do our research on an everyday basis.

## Installation

You can type

``` pip install PyMoosh ```

it should work !

## For specialists

PyMoosh is based on a scattering matrix formalism to solve Maxwell's equations in a multilayered structure. This makes PyMoosh unconditionally stable, allowing to explore even advanced properties of such multilayers, find poles and zeros of the scattering matrix (and thus guided modes), and many other things...


## References

If you use PyMoosh and if this is relevant, please cite the [paper associated with Moosh](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.100/)

```
@article{defrance2016moosh,
title={Moosh: A numerical swiss army knife for the optics of multilayers in octave/matlab},
author={Defrance, Josselin and Lema{\^\i}tre, Caroline and Ajib, Rabih and Benedicto, Jessica and Mallet, Emilien and Poll{\`e}s, R{\'e}mi and Plumey, Jean-Pierre and Mihailovic, Martine and Centeno, Emmanuel and Cirac{\`\i}, Cristian and others},
journal={Journal of Open Research Software},
volume={4},
number={1},
year={2016},
publisher={Ubiquity Press}
}
```

Even if PyMoosh is quite simple, this is a research-grade program. We actually do research with it. We've done cool things, like [comparing evolutionary algorithms and real evolution for the first time in history](https://www.nature.com/articles/s41598-020-68719-3).

## Contributors

Here is a list of contributors to PyMoosh (one way or another) so far:

* Pauline Bennet (@Ellawin)
* Peter Wiecha
* Demetrio Macias
* Anorld Capo-Chichi

and the contributors to the original Moosh program should not be forgotten : Josselin Defrance, Rémi Pollès, Fabien Krayzel, Paul-Henri Tichit, Jessica Benedicto mainly. Special thanks to Gérard Granet and Jean-Pierre Plumey.
