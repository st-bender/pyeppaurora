# PyEPPAurora

**Atmospheric ionization from particle precipitation**

[![builds](https://github.com/st-bender/pyeppaurora/actions/workflows/ci_build_and_test.yml/badge.svg?branch=master)](https://github.com/st-bender/pyeppaurora/actions/workflows/ci_build_and_test.yml)
[![docs](https://readthedocs.org/projects/pyeppaurora/badge/?version=latest)](https://pyeppaurora.readthedocs.io/en/latest/?badge=latest)
[![package](https://img.shields.io/pypi/v/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![wheel](https://img.shields.io/pypi/wheel/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![pyversions](https://img.shields.io/pypi/pyversions/eppaurora.svg?style=flat)](https://pypi.org/project/eppaurora)
[![codecov](https://codecov.io/gh/st-bender/pyeppaurora/badge.svg)](https://codecov.io/gh/st-bender/pyeppaurora)
[![coveralls](https://coveralls.io/repos/github/st-bender/pyeppaurora/badge.svg)](https://coveralls.io/github/st-bender/pyeppaurora)
[![scrutinizer](https://scrutinizer-ci.com/g/st-bender/pyeppaurora/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/st-bender/pyeppaurora/?branch=master)

[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.4298136.svg)](https://doi.org/10.5281/zenodo.4298136)

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
- `h5netcdf` - optional for the empirical models, install with `eppaurora[models]`
- `xarray` - optional for the empirical models, install with `eppaurora[models]`
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

### Energetic particle input in the atmosphere

This module include various parametrizations that describe the
energy dissipation of electrons and protons entering the middle
and upper atmosphere (see [References](#references) below).
The functions are correspondingly named

- `rr1987()` for the parametrization by Roble and Ridley, 1987,
- `fang2008()` for the parametrization described in Fang et al., 2008,
- `fang2010()`  for the parametrization described in Fang et al., 2010, and
- `fang2013()` for the proton parametrization by Fang et al., 2013

For example, they are called like this:

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

All functions need additional input for the background atmosphere,
the scale height and the mass density as described in the listed publications.
These can be obtained, for example, from the `nrlmsise00` module
<https://github.com/st-bender/pynrlmsise00>.
Profiles can then be calculated by passing the respective scale height
and density profiles in addition to the energy and flux.

`fang1020()` and `fang2013()` are for mono-energetic particles and to
obtain a realistic description, the results should be integrated
over the respective energy spectrum.
For this there are spectra functions available for Gaussian, Maxwellian,
and power-law distribution.

### Recombination rates

Some atmospheric recombination rates $\alpha$ are available within
`eppaurora.recombination` to convert the ionization rates $q$
to electron densities $n_e$ via $q = \alpha n_e^2$.
The recombination rates are parametrized according to altitude,
see [References](#references).

### Conductivity and conductance

Estimators for the Hall and Pedersen conductivity and conductance
are available via the `eppaurora.conductivity` module.
It contains the approximate solution "Robinson formula" for the conductances,
and the functions for the conductivities that need the electron density
and a model for the magnetic field,
see [References](#references).

For example the "Robinson" conductances for an average energy `en_avg` and flux `flx`
can be obtained by calling `SigmaH_robinson1987(<en_avg>, <flx>)`:
and `SigmaP_robinson1987(<en_avg>, <flx>)`:

```python
>>> import eppaurora as aur
>>> aur.SigmaH_robinson1987(10.0, 1.0)
10.985365619753862
>>> aur.SigmaP_robinson1987(10.0, 1.0)
3.4482758620689653

```

### Empirical ionization rate models

This package provides the coefficients and evaluation function
for the SSUSI-derived ionization rate model.
It is imported via `eppaurora.models` and the coefficients are
available through `ssusiq2023_coeffs()`.

The model itself can be evaluated with `ssusiq2023()`
for a certain geomagnetic latitude (gmlat),
magnetic local time (mlt), and altitude by providing the space-weather
coefficients of Kp, PC, Ap, and the 81-day averaged F10.7 fluxes as inputs:
`ssusiq2023(<gmlat>, <mlt>, <altitude>, [<list of index values>])`.
Note that this returns the natural logarithm of the ionization rate,
normalized to 1 cm⁻³ s⁻¹.

Instead of a list or `numpy`-array, an `xarray.DataArray` can be used for
the indices which will promote the coordinates to the result,
such as an extra time dimension.

```python
>>> from eppaurora.models import ssusiq2023
>>> ssusiq2023(65.0, 3.0, 100.0, [4.0, 10.0, 100.0, 157.0])  # doctest: +SKIP
<xarray.DataArray 'log_q' (dim_0: 1)>
array([18.80679417])
Coordinates:
    altitude  float32 100.0
    latitude  float32 66.6
    mlt       float32 3.0
Dimensions without coordinates: dim_0
Attributes:
    long_name:  natural logarithm of ionization rate
    units:      log(cm-3 s-1)

```

Various options to use different coefficients or to interpolate
to a finer grid exist, check the docstring of `ssusiq2023()`.
The geomagnetic indices can be obtained, for example,
from the `spaceweather` module
<https://github.com/st-bender/pyspaceweather>.

### Other

Basic class and method documentation is accessible via `pydoc`:

```sh
$ pydoc eppaurora
$ pydoc eppaurora.brems
$ pydoc eppaurora.conductivity
$ pydoc eppaurora.electrons
$ pydoc eppaurora.protons
$ pydoc eppaurora.recombination
$ pydoc eppaurora.models.ssusiq2023
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

### Conductivity and conductance

[9]: Brekke et al., J. Geophys. Res., 79(25), 3773--3790, Sept. 1974,
doi: [10.1029/JA079i025p03773](https://doi.org/10.1029/JA079i025p03773)  
[10]: Vickrey et al., J. Geophys. Res., 86(A1), 65--75, Jan. 1981,
doi: [10.1029/JA086iA01p00065](https://doi.org/10.1029/JA086iA01p00065)  
[11]: Robinson et al., J. Geophys. Res. Space Phys., 92(A3), 2565--2569, Mar. 1987,
doi: [10.1029/JA092iA03p02565](https://doi.org/10.1029/JA092iA03p02565)  

## License

This python interface is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
