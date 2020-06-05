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

```python
>>> import electronaurora as aur
>>> ion_prof = aur.rr1987(en, flux, scale_heights, rhos)

```

Basic class and method documentation is accessible via `pydoc`:

```sh
$ pydoc electronaurora
```

# License

This python interface is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
