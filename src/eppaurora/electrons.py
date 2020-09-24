# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric ionization rate parametrizations

Includes the atmospheric ionization rate parametrizations for auroral
and medium-energy electron precipitation, 100 eV--1 MeV [1]_, [2]_, and [3]_.

.. [1] Roble and Ridley, Ann. Geophys., 5A(6), 369--382, 1987
.. [2] Fang et al., J. Geophys. Res., 113, A09311, 2008
.. [3] Fang et al., Geophys. Res. Lett., 37, L22106, 2010
"""

import numpy as np
from numpy.polynomial.polynomial import polyval

from .spectra import pflux_maxwell

__all__ = [
	"rr1987",
	"rr1987_mod",
	"fang2008",
	"fang2010_mono",
	"fang2010_spec_int",
	"fang2010_maxw_int",
]

POLY_F2008 = [
	[ 3.49979e-1, -6.18200e-2, -4.08124e-2,  1.65414e-2],
	[ 5.85425e-1, -5.00793e-2,  5.69309e-2, -4.02491e-3],
	[ 1.69692e-1, -2.58981e-2,  1.96822e-2,  1.20505e-3],
	[-1.22271e-1, -1.15532e-2,  5.37951e-6,  1.20189e-3],
	[ 1.57018,     2.87896e-1, -4.14857e-1,  5.18158e-2],
	[ 8.83195e-1,  4.31402e-2, -8.33599e-2,  1.02515e-2],
	[ 1.90953,    -4.74704e-2, -1.80200e-1,  2.46652e-2],
	[-1.29566,    -2.10952e-1,  2.73106e-1, -2.92752e-2]
]

POLY_F2010 = [
	[ 1.24616E+0,  1.45903E+0, -2.42269E-1,  5.95459E-2],
	[ 2.23976E+0, -4.22918E-7,  1.36458E-2,  2.53332E-3],
	[ 1.41754E+0,  1.44597E-1,  1.70433E-2,  6.39717E-4],
	[ 2.48775E-1, -1.50890E-1,  6.30894E-9,  1.23707E-3],
	[-4.65119E-1, -1.05081E-1, -8.95701E-2,  1.22450E-2],
	[ 3.86019E-1,  1.75430E-3, -7.42960E-4,  4.60881E-4],
	[-6.45454E-1,  8.49555E-4, -4.28581E-2, -2.99302E-3],
	[ 9.48930E-1,  1.97385E-1, -2.50660E-3, -2.06938E-3]
]


def rr1987(energy, flux, scale_height, rho):
	"""Atmospheric electron energy dissipation Roble and Ridley, 1987 [#]_

	Equations (typo corrected) taken from Fang et al., 2008.

	Parameters
	----------
	energy: array_like (M,...)
		Characteristic energy E_0 [keV] of the Maxwellian distribution.
	flux: array_like (M,...)
		Integrated energy flux Q_0 [keV / cm² / s¹]
	scale_height: array_like (N,...)
		The atmospheric scale heights [cm].
	rho: array_like (N,...)
		The atmospheric mass density [g / cm³]

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	References
	----------

	.. [#] Roble and Ridley, Ann. Geophys., 5A(6), 369--382, 1987
	"""
	_c1 = 2.11685
	_c2 = 2.97035
	_c3 = 2.09710
	_c4 = 0.74054
	_c5 = 0.58795
	_c6 = 1.72746
	_c7 = 1.37459
	_c8 = 0.93296

	beta = (rho * scale_height / (4 * 1e-6))**(1 / 1.65)  # RR 1987, p. 371
	y = beta / energy  # Corrected in Fang et al. 2008 (4)
	f_y = (_c1 * (y**_c2) * np.exp(-_c3 * (y**_c4)) +
		_c5 * (y**_c6) * np.exp(-_c7 * (y**_c8)))
	# Corrected in Fang et al. 2008 (2)
	en_diss = 0.5 * flux / scale_height * f_y
	return en_diss


def rr1987_mod(energy, flux, scale_height, rho):
	"""Atmospheric electron energy dissipation Roble and Ridley, 1987 [#]_

	Equations (typo corrected) taken from Fang et al., 2008.
	Modified polynomial values to get closer to Fang et al., 2008,
	origin unknown.

	Parameters
	----------
	energy: array_like (M,...)
		Characteristic energy E_0 [keV] of the Maxwellian distribution.
	flux: array_like (M,...)
		Integrated energy flux Q_0 [keV / cm² / s¹]
	scale_height: array_like (N,...)
		The atmospheric scale heights [cm].
	rho: array_like (N,...)
		The atmospheric mass density [g / cm³]

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	References
	----------

	.. [#] Roble and Ridley, Ann. Geophys., 5A(6), 369--382, 1987
	"""
	# Modified polynomial, origin unknown
	_c1 = 3.233
	_c2 = 2.56588
	_c3 = 2.2541
	_c4 = 0.7297198
	_c5 = 1.106907
	_c6 = 1.71349
	_c7 = 1.8835444
	_c8 = 0.86472135

	# Fang et al., 2008, Eq. (4)
	y = (rho * scale_height / (4.6 * 1e-6))**(1 / 1.65) / energy
	f_y = (_c1 * (y**_c2) * np.exp(-_c3 * (y**_c4)) +
		_c5 * (y**_c6) * np.exp(-_c7 * (y**_c8)))
	# energy dissipated [keV]
	en_diss = 0.5 * flux / scale_height * f_y
	return en_diss


def _fang_f_y(_c, _y):
	"""Polynomial evaluation helper

	Fang et al., 2008, Eq. (6), Fang et al., 2010 Eq. (4)
	"""
	ret = (
		_c[0] * (_y**_c[1]) * np.exp(-_c[2] * (_y**_c[3])) +
		_c[4] * (_y**_c[5]) * np.exp(-_c[6] * (_y**_c[7]))
	)
	return ret


def fang2008(energy, flux, scale_height, rho, pij=POLY_F2008):
	"""Atmospheric electron energy dissipation from Fang et al., 2008

	Ionization profile parametrization as derived in Fang et al., 2008 [#]_.

	Parameters
	----------
	energy: array_like (M,...)
		Characteristic energy E_0 [keV] of the Maxwellian distribution.
	flux: array_like (M,...)
		Integrated energy flux Q_0 [keV / cm² / s¹]
	scale_height: array_like (N,...)
		The atmospheric scale height(s) [cm].
	rho: array_like (N,...)
		The atmospheric densities [g / cm³], corresponding to the scale heights.

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	References
	----------

	.. [#] Fang et al., J. Geophys. Res., 113, A09311, 2008, doi: 10.1029/2008JA013384
	"""
	pij = np.asarray(pij)
	# Fang et al., 2008, Eq. (7)
	_cs = np.exp(polyval(np.log(energy), pij.T))
	# Fang et al., 2008, Eq. (4)
	y = (rho * scale_height / (4e-6))**(1 / 1.65) / energy
	f_y = _fang_f_y(_cs, y)
	# Fang et al., 2008, Eq. (2)
	en_diss = 0.5 * f_y * flux / scale_height
	return en_diss


def fang2010_mono(energy, flux, scale_height, rho, pij=POLY_F2010):
	r"""Atmospheric electron energy dissipation from Fang et al., 2010

	Parametrization for mono-energetic electrons [#]_.

	Parameters
	----------
	energy: array_like (M,...)
		Energy E_0 of the mono-energetic electron beam [keV].
	flux: array_like (M,...)
		Energy flux Q_0 of the mono-energetic electron beam [keV / cm² / s¹].
	scale_height: array_like (N,...)
		The atmospheric scale heights [cm].
	rho: array_like (N,...)
		The atmospheric mass densities [g / cm³], corresponding to the scale heights.

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	References
	----------

	.. [#] Fang et al., Geophys. Res. Lett., 37, L22106, 2010, doi: 10.1029/2010GL045406
	"""
	pij = np.asarray(pij)
	# Fang et al., 2010, Eq. (5)
	_cs = np.exp(polyval(np.log(energy), pij.T))
	# Fang et al., 2010, Eq. (1)
	y = 2. / energy * (rho * scale_height / (6e-6))**(0.7)
	f_y = _fang_f_y(_cs, y)
	# Fang et al., 2008, Eq. (2)
	en_diss = f_y * flux / scale_height
	return en_diss


def fang2010_spec_int(ens, dfluxes, scale_height, rho, pij=POLY_F2010, axis=-1):
	r"""Integrate over a given energy spectrum

	Integrates over the mono-energetic parametrization `q` from Fang et al., 2010
	using the given differential particle spectrum `phi`:

	:math:`\int_\text{spec} \phi(E) q(E, Q) E \text{d}E`

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

	Returns
	-------
	en_diss: array_like (N)
		The dissipated energy profiles [keV].

	See Also
	--------
	fang2010_mono
	"""
	ediss_f10 = fang2010_mono(
		ens[None, None, :],
		dfluxes,
		scale_height[..., None],
		rho[..., None],
		pij=pij,
	)
	return np.trapz(ediss_f10 * ens, ens, axis=axis)


def fang2010_maxw_int(energy, flux, scale_height, rho, bounds=(0.1, 300.), nstep=128, pij=POLY_F2010):
	"""Integrate Fang et al., 2010 over a Maxwellian spectrum

	Integrates the mono-energetic parametrization from Fang et al., 2010 [#]_
	over a Maxwellian spectrum with characteristic energy `energy` and
	total energy flux `flux`.

	Parameters
	----------
	energy: float or array_like (M,...)
		Characteristic energy E_0 [keV] of the Maxwellian distribution.
	flux: float or array_like (M,...)
		Integrated energy flux Q_0 [keV / cm² / s¹]
	scale_height: float or array_like (N,...)
		The atmospheric scale heights [cm].
	rho: float or array_like (N,...)
		The atmospheric mass density [g / cm³]
	bounds: tuple, optional
		(min, max) [keV] of the integration range to integrate the Maxwellian.
		Make sure that this is appropriate to encompass the spectrum.
		Default: (0.1, 300.)
	nsteps: int, optional
		Number of integration steps, default: 128.

	Returns
	-------
	en_diss: array_like (M,N)
		The dissipated energy profiles [keV].

	See Also
	--------
	fang2010_mono, fang2010_spec_int, maxwell_pflux
	"""
	energy = np.asarray(energy)
	flux = np.asarray(flux)
	bounds_l10 = np.log10(bounds)
	ens = np.logspace(*bounds_l10, num=nstep)
	ensd = np.reshape(ens, (-1,) + (1,) * energy.ndim)
	dflux = flux.T * pflux_maxwell(ensd, energy.T)
	return fang2010_spec_int(ens, dflux.T, scale_height, rho, pij=pij, axis=-1)
