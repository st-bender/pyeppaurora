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
	"ediss_spec_int",
	"ediss_specfun_int",
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


def ediss_spec_int(
	ens,
	dfluxes,
	scale_height,
	rho,
	func,
	axis=-1,
	func_kws=None,
):
	r"""Integrate over a given energy spectrum

	Integrates a mono-energetic parametrization `q`, e.g. from Fang et al., 2010
	using the given differential particle spectrum `phi`:

	:math:`\int_\text{spec} \phi(E) q(E, Q) E \text{d}E`

	This function uses the differential spectrum evaluated at the given energy bins.

	Parameters
	----------
	ens: array_like (M,...)
		Central (bin) energies of the spectrum
	dfluxes: array_like (M,...)
		Differential particle fluxes in the given bins
	scale_height: array_like (N,...)
		The atmospheric scale heights
	rho: array_like (N,...)
		The atmospheric densities, corresponding to the
		scale heights.
	func: callable
		Mono-energetic energy dissipation function to integrate.
	axis: int, optional
		The axis to use for integration, default: -1 (last axis).
	func_kws: dict-like, optional
		Optional keyword arguments to pass to the mono-energetic
		energy dissipation function. Default: `None`

	Returns
	-------
	en_diss: array_like (N)
		The dissipated energy profiles [keV].

	See Also
	--------
	ediss_specfun_int
	"""
	ens = np.atleast_1d(ens)
	dfluxes = np.atleast_1d(dfluxes)
	scale_height = np.atleast_1d(scale_height)
	rho = np.atleast_1d(rho)
	func_kws = func_kws or dict()
	ediss = func(
		ens[None, None, :],
		dfluxes,
		scale_height[..., None],
		rho[..., None],
		**func_kws,
	)
	return np.trapz(ediss * ens, ens, axis=axis)


def ediss_specfun_int(
	energy,
	flux,
	scale_height,
	rho,
	ediss_func,
	ediss_kws=None,
	bounds=(0.1, 300.),
	nstep=128,
	spec_fun=pflux_maxwell,
	spec_kws=None,
):
	"""Integrate mono-energetic parametrization over a spectrum

	Integrates the mono-energetic parametrization over a spectrum given by a
	functional dependence with characteristic energy `energy` and total energy
	flux `flux`.

	Parameters
	----------
	energy: float or array_like (M,...)
		Characteristic energy E_0 [keV] of the spectral distribution.
	flux: float or array_like (M,...)
		Integrated energy flux Q_0 [keV / cm² / s¹]
	scale_height: float or array_like (N,...)
		The atmospheric scale heights [cm].
	rho: float or array_like (N,...)
		The atmospheric mass density [g / cm³]
	ediss_func: callable
		Mono-energetic energy dissipation function to integrate.
	ediss_kws: dict-like, optional
		Optional keyword arguments to pass to the mono-energetic
		energy dissipation function. Default: `None`
	bounds: tuple, optional
		(min, max) [keV] of the integration range to integrate the Maxwellian.
		Make sure that this is appropriate to encompass the spectrum.
		Default: (0.1, 300.)
	nsteps: int, optional
		Number of integration steps, default: 128.
	spec_func: callable, optional, default :func:`pflux_maxwell`
		Spectral shape function, choices are:
		* :func:`pflux_exp` for a exponential spectrum
		* :func:`pflux_gaussian` for a Gaussian shaped spectrum
		* :func:`pflux_maxwell` for a Maxwellian shaped spectrum
		* :func:`pflux_pow` for a power-law
	spec_kws: dict-like, optional
		Optional keyword arguments to pass to the spectral function
		Default: `None`

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	See Also
	--------
	ediss_spec_int
	"""
	energy = np.asarray(energy)
	flux = np.asarray(flux)
	bounds_l10 = np.log10(bounds)
	ens = np.logspace(*bounds_l10, num=nstep)
	ensd = np.reshape(ens, (-1,) + (1,) * energy.ndim)
	spec_kws = spec_kws or dict()
	# "overwrite" the characteristic energy
	spec_kws["en_0"] = energy.T
	dflux = flux.T * spec_fun(ensd, **spec_kws)
	return ediss_spec_int(
		ens, dflux.T, scale_height, rho, ediss_func,
		axis=-1, func_kws=ediss_kws,
	)
