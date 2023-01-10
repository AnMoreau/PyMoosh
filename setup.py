from setuptools import setup, find_packages

setup(
    name='PyMoosh',
    version='2.1',
    license='GNU GENERAL PUBLIC LICENSE 3.0',
    author="Antoine Moreau",
    author_email='antoine.moreau@uca.fr',
    packages=find_packages('code'),
    package_dir={'': 'code'},
    url='https://github.com/AnMoreau/PyMoosh',
    keywords=['Moosh','Maxwell','Optics','Multilayers','Plasmonics','Photovoltaics'],
    install_requires=[
          'numpy','matplotlib','scipy'
      ],

)
