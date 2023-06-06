# PyMoosh

## About PyMoosh

PyMoosh is a swiss knife for the study of multilayered structures from an optical point of view, written in Python. 

PyMoosh is now much more advanced than Moosh, the original octave/matlab program we used in the past. Importantly, the use of Moosh is illustrated by many Jupyter notebooks (collabs are coming) and even more are planned. PyMoosh can be used for teaching or research purposes. It is especially written to be stable and quick, for its use in an optimization framework for instance.

![What Moosh (green) can do...](field.png)

## Installation

You can do something as simple as 

``` pip install pymoosh ```

## For specialists

PyMoosh is based on a scattering matrix formalism to solve Maxwell's equations in a multilayered structure. This makes PyMoosh unconditionally stable, allowing to explore even advanced properties of such multilayers, find poles and zeros of the scattering matrix (and thus guided modes), and many other things... We have included all the known kind of formalism to solve Maxwell's equations in such structures (admittance formalism, Abeles matrices, transfer matrices...).

## References

If you use PyMoosh and if this is relevant, please cite the [paper associated with Moosh](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.100/). Another paper is on its way, hopefully...

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
* Denis Langevin (@Milloupe)
* Olivier Teytaud (@teytaud)
* Demetrio Macias
* Anorld Capo-Chichi

and the contributors to the original Moosh program should not be forgotten : Josselin Defrance, Rémi Pollès, Fabien Krayzel, Paul-Henri Tichit, Jessica Benedicto mainly, but David R. Smith and Cristian Ciraci too ! Special thanks to Gérard Granet and Jean-Pierre Plumey.
