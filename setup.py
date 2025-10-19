"""Setup/build/install script for ZERNIPAX."""

import os

import versioneer
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="zernipax",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=(
        "Includes functions to generate Zernike modes and Calculate Zernike "
        "polynomials and their derivatives using JAX."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PlasmaControl/ZERNIPAX/",
    author=(
        "Yigit Gunsur Elmacioglu, Rory Conlin, Dario Panici, "
        "Daniel Dudt, Egemen Kolemen"
    ),
    author_email="PlasmaControl@princeton.edu",
    license="MIT",
    keywords="zernike polynomials, optics, astrophysics, spectral "
    + "simulation, basis, orthogonal polynomials, parallel computing, "
    + "JAX",
    packages=find_packages(exclude=["tests", "notebooks", "figures"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
