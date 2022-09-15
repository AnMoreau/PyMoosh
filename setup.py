from setuptools import setup, find_packages


setup(
    name='pyMoosh',
    version='1.0',
    license='TBD',
    author="Antoine Moreau",
    author_email='antoine.moreau@uca.fr',
    packages=find_packages('code'),
    package_dir={'': 'code'},
    url='https://github.com/AnMoreau/PyMoosh',
    keywords=['Moosh','Maxwell','Scattering','Plasmons'],
    install_requires=[
          'numpy',
      ],

)
