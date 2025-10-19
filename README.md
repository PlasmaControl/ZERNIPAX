# Zernipax
A python library to calculate Zernike Polynomials fast and accurately using JAX.

To create conda environment:
```
conda create --name zernipax-env 'python>=3.8, <=3.12'
conda activate zernipax-env
pip install -r requirements.txt
```

or, if you want to contribute
```
pip install -r dev-requirement.txt
```

## GPU Support
JAX installation for different systems may vary. Please refer to [their documentation](https://docs.jax.dev/en/latest/installation.html#installation) for details. Usually, you need to run something like following,

```
pip install jax[cuda12]
```

## Citation
If you use this repository in you projects, please cite it as:
```
@article{ELMACIOGLU2025129534,
title = {ZERNIPAX: A fast and accurate Zernike polynomial calculator in Python},
journal = {Applied Mathematics and Computation},
volume = {505},
pages = {129534},
year = {2025},
issn = {0096-3003},
doi = {https://doi.org/10.1016/j.amc.2025.129534},
url = {https://www.sciencedirect.com/science/article/pii/S0096300325002607},
author = {Yigit Gunsur Elmacioglu and Rory Conlin and Daniel W. Dudt and Dario Panici and Egemen Kolemen},
keywords = {Zernike polynomials, Optics, Astrophysics, Spectral simulations, Python, JAX, CPU/GPU computing},
abstract = {Zernike polynomials serve as an orthogonal basis on the unit disc, and have proven to be effective in optics simulations, astrophysics, and more recently in plasma simulations. Unlike Bessel functions, Zernike polynomials are inherently finite and smooth at the disc center (r=0), ensuring continuous differentiability along the axis. This property makes them particularly suitable for simulations, requiring no additional handling at the origin. We developed ZERNIPAX, an open-source Python package capable of utilizing CPU/GPUs, leveraging Google's JAX package and available on GitHub as well as the Python software repository PyPI. Our implementation of the recursion relation between Jacobi polynomials significantly improves computation time compared to alternative methods by use of parallel computing while still performing more accurately for high-mode numbers.}
}
```
