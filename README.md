# PyEPPAurora

**Atmospheric ionization from particle precipitation**

[![builds](https://travis-ci.com/st-bender/pyeppaurora.svg?branch=master)](https://travis-ci.com/st-bender/pyeppaurora)
[![docs](https://readthedocs.org/projects/pyeppaurora/badge/?version=latest)](https://pyeppaurora.readthedocs.io/en/latest/?badge=latest)
[![package](https://img.shields.io/pypi/v/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![wheel](https://img.shields.io/pypi/wheel/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![pyversions](https://img.shields.io/pypi/pyversions/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![codecov](https://codecov.io/gh/st-bender/pyeppaurora/badge.svg)](https://codecov.io/gh/st-bender/pyeppaurora)
[![coveralls](https://coveralls.io/repos/github/st-bender/pyeppaurora/badge.svg)](https://coveralls.io/github/st-bender/pyeppaurora)
[![scrutinizer](https://scrutinizer-ci.com/g/st-bender/pyeppaurora/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/st-bender/pyeppaurora/?branch=master)

Bundles some of the parametrizations for middle and upper atmospheric
ionization and recombination rates for precipitating
auroral and radiation-belt electrons as well as protons.
Includes also some recombination rate parametrizations to convert
the ionization rates to electron densities in the upper atmosphere.
See [References](#references) for a list of included parametrizations.

:warning: This package is in **beta** stage, that is, it works for the most part
and the interface should not change (much) in future versions.

Documentation is available at <https://pyeppaurora.readthedocs.io>.

## Install

### Requirements

- `numpy` - required
- `scipy` - required for 2-D interpolation
- `pytest` - optional, for testing

### eppaurora

An installable `pip` package called `eppaurora` is available from the
main package repository, it can be installed with:
```sh
$ pip install eppaurora
```
The latest development version can be installed
with [`pip`](https://pip.pypa.io) directly from github
(see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):

```sh
$ pip install [-e] git+https://github.com/st-bender/pyeppaurora.git
```

The other option is to use a local clone:

```sh
$ git clone https://github.com/st-bender/pyeppaurora.git
$ cd pyeppaurora
```
and then using `pip` (optionally using `-e`, see
<https://pip.pypa.io/en/stable/reference/pip_install/#install-editable>):

```sh
$ pip install [-e] .
```

or using `setup.py`:

```sh
$ python setup.py install
```

Optionally, test the correct function of the module with

```sh
$ py.test [-v]
```

or even including the [doctests](https://docs.python.org/library/doctest.html)
in this document:

```sh
$ py.test [-v] --doctest-glob='*.md'
```

## Usage

The python module itself is named `eppaurora` and is imported as usual.

All functions should be `numpy`-compatible and work with scalars
and appropriately shaped arrays.

```python
>>> import eppaurora as aur
>>> ediss = aur.rr1987(1., 1., 8e5, 5e-10)
>>> ediss
3.3693621076457477e-10
>>> import numpy as np
>>> energies = np.logspace(-1, 2, 4)
>>> fluxes = np.ones_like(energies)
>>> # ca. 100, 150, 200 km
>>> scale_heights = np.array([6e5, 27e5, 40e5])
>>> rhos = np.array([5e-10, 1.7e-12, 2.6e-13])
>>> # energy dissipation "profiles"
>>> # broadcast to the right shape
>>> ediss_prof = aur.fang2008(
... 	energies[None, :], fluxes[None, :],
... 	scale_heights[:, None], rhos[:, None]
... )
>>> ediss_prof
array([[1.37708081e-49, 3.04153876e-09, 4.44256875e-07, 2.52699970e-08],
       [1.60060833e-09, 8.63248169e-08, 3.64564419e-09, 1.62591310e-10],
       [5.19369952e-08, 2.34089350e-08, 5.17379303e-10, 3.19504690e-11]])

```

Basic class and method documentation is accessible via `pydoc`:

```sh
$ pydoc eppaurora
$ pydoc eppaurora.brems
$ pydoc eppaurora.electrons
$ pydoc eppaurora.protons
$ pydoc eppaurora.recombination
```

## References

### Electron ionization

[1]: Roble and Ridley, Ann. Geophys., 5A(6), 369--382, 1987  
[2]: Fang et al., J. Geophys. Res. Space Phys., 113, A09311, 2008,
doi: [10.1029/2008JA013384](https://doi.org/10.1029/2008JA013384)  
[3]: Fang et al., Geophys. Res. Lett., 37, L22106, 2010,
doi: [10.1029/2010GL045406](https://doi.org/10.1029/2010GL045406)  

### Ionization by secondary electrons from bremsstrahlung

[4]: Berger et al., Journal of Atmospheric and Terrestrial Physics,
Volume 36, Issue 4, 591--617, April 1974,
doi: [10.1016/0021-9169(74)90085-3](https://doi.org/10.1016/0021-9169%2874%2990085-3)

### Proton ionization

[5]: Fang et al., J. Geophys. Res. Space Phys., 118, 5369--5378, 2013,
doi: [10.1002/jgra.50484](https://doi.org/10.1002/jgra.50484)

### Recombination rates

[6]: Vickrey et al., J. Geophys. Res. Space Phys., 87, A7, 5184--5196,
doi: [10.1029/ja087ia07p05184](https://doi.org/10.1029/ja087ia07p05184)  
[7]: Gledhill, Radio Sci., 21, 3, 399-408,
doi: [10.1029/rs021i003p00399](https://doi.org/10.1029/rs021i003p00399)  
[8]: https://ssusi.jhuapl.edu/data_algorithms

## License

This python interface is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
