# Atmospheric ionization from electron precipitation

[![builds](https://travis-ci.com/st-bender/electronaurora.svg?branch=master)](https://travis-ci.com/st-bender/electronaurora)
[![codecov](https://codecov.io/gh/st-bender/electronaurora/badge.svg)](https://codecov.io/gh/st-bender/electronaurora)
[![coveralls](https://coveralls.io/repos/github/st-bender/electronaurora/badge.svg)](https://coveralls.io/github/st-bender/electronaurora)
[![scrutinizer](https://scrutinizer-ci.com/g/st-bender/electronaurora/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/st-bender/electronaurora/?branch=master)

Calculates atmospheric ionization profiles from electron precipitation
using different parametrizations: Roble and Ridley, 1987 [1],
Fang et al., 2008 [2], and Fang et al., 2010 [3].

## Install

### Requirements

- `numpy` - required
- `pytest` - optional, for testing

### electronaurora

As binary package support is limited, electronaurora can be installed
with [`pip`](https://pip.pypa.io) directly from github
(see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):

```sh
$ pip install [-e] git+https://github.com/st-bender/electronaurora.git
```

The other option is to use a local clone:

```sh
$ git clone https://github.com/st-bender/electronaurora.git
$ cd electronaurora
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

The python module itself is named `electronaurora` and is imported as usual.

All functions should be `numpy`-compatible and work with scalars
and appropriately shaped arrays.

```python
>>> import electronaurora as aur
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
$ pydoc electronaurora
```

# References

[1]: Roble and Ridley, Ann. Geophys., 5A(6), 369--382, 1987
[2]: Fang et al., J. Geophys. Res., 113, A09311, 2008, doi: 10.1029/2008JA013384
[3]: Fang et al., Geophys. Res. Lett., 37, L22106, 2010, doi: 10.1029/2010GL045406

# License

This python interface is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
