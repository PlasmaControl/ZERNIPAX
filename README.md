# Zernipax
A python library to calculate Zernike Polynomials fast and accurately using JAX. Available on `PyPI`,

```
pip install zernipax
```

## GPU Support
JAX installation for different systems may vary. Please refer to [their documentation](https://docs.jax.dev/en/latest/installation.html#installation) for details. Usually, you need to run something like following,

```
pip install jax[cuda12]
```

## Citation
If you use this repository in your projects, please cite it as:
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
    author = {Yigit Gunsur Elmacioglu, Rory Conlin, Daniel W. Dudt, Dario Panici and Egemen Kolemen},
}
```
