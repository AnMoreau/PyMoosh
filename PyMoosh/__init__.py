# encoding: utf-8
#
#Copyright (C) 2022-2023, Antoine Moreau
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
    PyMoosh - a scattering matrix formalism to solve Maxwell's equations
    in a multilayered structure. This makes PyMoosh unconditionally stable,
    allowing to explore even advanced properties of such multilayers,
    find poles and zeros of the scattering matrix (and thus guided modes),
    and many other things...

"""
__name__ = 'PyMoosh'
__version__ = '2.2'
__date__ = "01/20/2023"   # MM/DD/YYY
__author__ = 'Antoine Moreau'




## make accessible everything from `core` directly from the PyMoosh base package
from PyMoosh.core import *
