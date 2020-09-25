# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Particle precipitation spectra

Includes variants describing a normalized particle flux,
as well as variants describing a normalized energy flux.
"""

import numpy as np

__all__ = [
	"exp_general",
	"gaussian_general",
	"maxwell_general",
	"pow_general",
	"pflux_exp",
	"pflux_gaussian",
	"pflux_maxwell",
	"pflux_pow",
]


# General normalized spectra, standard distributions
def exp_general(en, en_0=10.):
	r"""Exponential number flux spectrum as in Aksnes et al., 2006 [0]

	Defined according to Aksnes et al., JGR 2006, Eq. (1),
	normalized to unit number flux, i.e.
	:math:`\int_0^\infty \phi(E) \text{d}E = 1`.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV] of the distribution.
		Default: 10 keV

	Returns
	-------
	phi: float or array_like (N,)
		Normalized differential hemispherical number flux at `en` in [keV-1 cm-2 s-1]
		([keV] or scaled by 1 keV-2 cm-2 s-1, e.g. ).
	"""
	return 1. / en_0 * np.exp(-en / en_0)


def gaussian_general(en, en_0=10., w=1.):
	r"""Gaussian number flux spectrum as in Fang2008 [1]

	Standard normal distribution with mu = en_0 and sigma = w / sqrt(2)
	for use in Fang et al., JGR 2008, Eq. (1).
	Almost normalized to unit number flux, i.e.
	:math:`\int_0^\infty \phi(E) \text{d}E = 1`
	(ignoring the negative tail).

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV
	w: float, optional
		Width of the Gaussian distribution, in [keV].

	Returns
	-------
	phi: float or array_like (N,)
		Normalized differential hemispherical number flux at `en` in [keV-1 cm-2 s-1]
		([keV] or scaled by 1 keV-2 cm-2 s-1, e.g.).
	"""
	return 1. / np.sqrt(np.pi * w**2) * np.exp(-(en - en_0)**2 / w**2)


def maxwell_general(en, en_0=10.):
	r"""Maxwell number flux spectrum as in Fang2008 [1]

	Defined in Fang et al., JGR 2008, Eq. (1),
	normalized to :math:`\int_0^\infty \phi(E) \text{d}E = 1`.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV

	Returns
	-------
	phi: float or array_like (N,)
		Normalized differential hemispherical number flux at `en` in [keV-1 cm-2 s-1]
		([keV] or scaled by 1 keV-2 cm-2 s-1, e.g.).
	"""
	return en / en_0**2 * np.exp(-en / en_0)


def pow_general(en, en_0=10., gamma=-3., het=True):
	r"""Power-law number flux spectrum as in Strickland1993 [3]

	Defined e.g. in Strickland et al., 1993,
	normalized to unit particle flux:
	:math:`\int_{E_0}^\infty \phi(E) \text{d}E = 1`
	for the high-energy tail version, and
	:math:`\int_0^{E_0} \phi(E) \text{d}E = 1`
	for the low-energy tail version.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV
	gamma: float, optional
		Exponent of the power-law distribution, in [keV].
	het: bool, optional
		Return a high-energy tail (het, default: true) for en > en_0,
		or low-energy tail (false) for en < en_0.
		Adjusts the normalization accordingly.

	Returns
	-------
	phi: float or array_like (N,)
		Normalized differential hemispherical number flux at `en` in [keV-1 cm-2 s-1]
		([keV] or scaled by 1 keV-2 cm-2 s-1, e.g.).
	"""
	spec = (gamma + 1) / en_0 * (en / en_0)**gamma
	if het:
		spec[en < en_0] = 0.
		return -spec
	spec[en > en_0] = 0.
	return spec


def pflux_exp(en, en_0=10.):
	r"""Exponential particle flux spectrum

	Normalized to unit energy flux:
	:math:`\int_0^\infty \phi(E) E \text{d}E = 1`.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV.

	Returns
	-------
	phi: float or array_like (N,)
		Hemispherical differential particle flux at `en` in [keV-1 cm-2 s-1]
		([kev-2] scaled by unit energy flux).
	"""
	return exp_general(en, en_0=en_0) / en_0


def pflux_gaussian(en, en_0=10., w=1):
	r"""Gaussian particle flux spectrum

	Defined in Fang et al., JGR 2008, Eq. (1).

	Normalized to :math:`\int_0^\infty \phi(E) E \text{d}E = 1`.
	(ignoring the negative tail).
	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV.

	Returns
	-------
	phi: float or array_like (N,)
		Hemispherical differential particle flux at `en` in [keV-1 cm-2 s-1]
		([kev-2] scaled by unit energy flux).
	"""
	return gaussian_general(en, en_0=en_0, w=w) / en_0


def pflux_maxwell(en, en_0=10.):
	r"""Maxwell particle flux spectrum as in Fang2008 [1]

	Defined in Fang et al., JGR 2008, Eq. (1).
	The total precipitating energy flux is fixed to 1 keV cm-2 s-1,
	multiply by Q_0 [keV cm-2 s-1] to scale the particle flux.

	Normalized to :math:`\int_0^\infty \phi(E) E \text{d}E = 1`.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV.

	Returns
	-------
	phi: float or array_like (N,)
		Hemispherical differential particle flux at `en` in [keV-1 cm-2 s-1]
		([kev-2] scaled by unit energy flux).
	"""
	return 0.5 / en_0 * maxwell_general(en, en_0)


def pflux_pow(en, en_0=10., gamma=-3., het=True):
	r"""Power-law particle flux spectrum

	Defined e.g. in Strickland et al., 1993.
	Normalized to :math:`\int_{E_0}^\infty \phi(E) E \text{d}E = 1`
	for the high-energy tail version, and to
	:math:`\int_0^{E_0} \phi(E) E \text{d}E = 1`
	for the low-energy tail version.

	Parameters
	----------
	en: float or array_like (N,)
		Energy in [keV]
	en_0: float, optional
		Characteristic energy in [keV], i.e. mode of the distribution.
		Default: 10 keV
	gamma: float, optional
		Exponent of the power-law distribution, in [keV].
	het: bool, optional (default True)
		Return a high-energy tail (true) for en > en_0,
		or low-energy tail (false) for en < en_0.
		Adjusts the normalization accordingly.

	Returns
	-------
	phi: float or array_like (N,)
		Hemispherical differential particle flux at `en` in [keV-1 cm-2 s-1]
		([kev-2] scaled by unit energy flux).
	"""
	return (gamma + 2) / (gamma + 1) / en_0 * pow_general(en, en_0=en_0, gamma=gamma, het=het)
