import re
import os
from setuptools import setup, find_packages
import sys


# =============================================================================
# helper functions to extract meta-info from package
# =============================================================================
def read_version_file(*parts):
    return open(os.path.join(*parts), 'r').read()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def find_version(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def find_name(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__name__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find name string.")

def find_author(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find author string.")


# =============================================================================
# setup    
# =============================================================================
if ((sys.version_info[0] >= 3 and sys.version_info[1] < 10) or sys.version_info[0] < 3):
    requires = [
          'numpy','matplotlib','scipy'
      ]
else:
    requires=[
          'numpy','matplotlib','scipy','refractiveindex'
      ]


setup(
    name = find_name("PyMoosh", "__init__.py"),
    version = find_version("PyMoosh", "__init__.py"),
    author = find_author("PyMoosh", "__init__.py"),
    author_email='antoine.moreau@uca.fr',
    license='MIT',
    packages=['PyMoosh'],
    include_package_data=True,   # add data folder containing material dispersions
    description = ("A scattering matrix formalism to solve Maxwell's equations " + 
                   "in a multilayered structure."),
    long_description=read('README.md'),
    url='https://github.com/AnMoreau/PyMoosh',
    keywords=['Moosh','Maxwell','Optics','Multilayers','Plasmonics','Photovoltaics'],
    install_requires=requires,

)
