# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric bremsstrahlung ionization parametrization [1]_

.. [1] Berger, M.J., Seltzer, S.M., Maeda, K.,
	Some new results on electron transport in the atmosphere,
	Journal of Atmospheric and Terrestrial Physics, v36, i4, pp. 591--617,
	April 1974,
	doi: 10.1016/0021-9169(74)90085-3
"""

import numpy as np
from scipy import interpolate

__all__ = ["berger1974"]

E_BR = [2., 5., 10., 20., 50., 100., 200., 500., 1000., 2000.]

Z_BR = [
	2e-6, 4e-6, 8e-6,
	2e-5, 4e-5, 8e-5,
	2e-4, 4e-4, 8e-4,
	2e-3, 4e-3, 8e-3,
	2e-2, 4e-2, 8e-2,
	2e-1, 4e-1,
]

A_BR = [
	[np.nan] * 9 + [8.6e-6, 5.1e-7] + [np.nan] * 6,
	[np.nan] * 8 + [7.2e-4, 1.4e-4, 3.3e-5, 5.2e-6, 1.1e-7] + [np.nan] * 4,
	[np.nan] * 7 + [2.0e-3, 8.8e-4, 2.7e-4, 9.4e-5, 2.8e-5, 3.0e-6, 3.8e-7, 3.9e-8] + [np.nan] * 2,
	[np.nan] * 6 + [3.4e-3, 1.7e-3, 8.0e-4, 3.0e-4, 1.3e-4, 5.7e-5, 1.5e-5, 3.3e-6, 5.7e-7, 2.9e-8] + [np.nan],
	[np.nan] * 5 + [7.3e-3, 2.2e-3, 1.1e-3, 5.7e-4, 2.3e-4, 1.1e-4, 5.4e-5, 2.0e-5, 8.3e-6, 2.6e-6, 3.1e-7, 2.8e-8],
	[np.nan] * 4 + [1.1e-2, 7.9e-3, 1.7e-3, 8.4e-4, 4.4e-4, 1.9e-4, 1.0e-4, 5.4e-5, 2.3e-5, 1.1e-5, 3.9e-6, 5.4e-7, 2.4e-8],
	[np.nan] * 3 + [5.0e-3, 5.3e-3, 5.0e-3, 1.8e-3, 6.5e-4, 3.3e-4, 1.5e-4, 8.8e-5, 5.2e-5, 2.4e-5, 1.1e-5, 3.8e-6, 1.7e-7, 5.9e-9],
	[np.nan] * 2 + [2.2e-3, 2.4e-3, 2.5e-3, 2.5e-3, 1.6e-3, 5.2e-4, 2.8e-4, 1.5e-4, 9.9e-5, 6.5e-5, 2.8e-5, 9.5e-6, 1.5e-6, 6.0e-9] + [np.nan],
	[np.nan] + [1.0e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.2e-3, 5.5e-4, 3.2e-4, 1.9e-4, 1.3e-4, 8.3e-5, 2.8e-5, 5.5e-6, 2.4e-7] + [np.nan] * 2,
	[6.8e-4, 7.6e-4, 8.4e-4, 9.3e-4, 9.9e-4, 1.1e-3, 1.1e-3, 6.8e-4, 4.5e-4, 2.9e-4, 1.9e-4, 9.2e-5, 1.7e-5, 1.2e-6] + [np.nan] * 3,
]


def berger1974(
	energy, flux,
	scale_height, rho,
	ens=E_BR, zm_p_en=Z_BR, coeffs=A_BR,
	fillna=None, log3=True,
	rbf="multiquadric",
):
	"""Bremsstrahlung ionization by secondary electrons

	Formulae and parameters as described in [2]_.

	By default, the `log(coefficients)` are interpolated wrt. `log(energy)`
	and `log(zm)` using :class:`scipy.interpolated.Rbf`.
	The default "multiquadric" should work fine, if not consider
	using "thin-plate" splines.

	.. [2] Berger, M.J., Seltzer, S.M., Maeda, K.,
		Some new results on electron transport in the atmosphere,
		Journal of Atmospheric and Terrestrial Physics, v36, i4, pp. 591--617,
		April 1974,
		doi: 10.1016/0021-9169(74)90085-3

	Parameters
	----------
	energy: array_like (M, ...)
		Energy E_0 of the mono-energetic electron beam [keV].
		A scalar (0-D) value is promoted to 1-D with one element.
	flux: array_like (M,...)
		Energy flux Q_0 of the mono-energetic electron beam [keV / cm² / s¹].
	scale_height: array_like (N, ...)
		The atmospheric scale heights [cm].
	rho: array_like (N, ...)
		The atmospheric mass density [g / cm³].
	ens: array_like (I,), optional
		The energies (one axis) of the coefficient array,
		used to interpolate the coefficients to `energy`.
		Defaults to the Berger [2]_ coefficients.
	zm_p_en: array_like (J,), optional
		The atmospheric depth (the other axis) of the coefficient array,
		used to interpolate the coefficients to `z` = `scale_height` * `rhos`.
		Defaults to the Berger [2]_ coefficients.
	coeffs: array_like, (J, I), optional
		The bremsstrahlung energy dissipation coefficients.
		Defaults to the Berger [2]_ coefficients.
	fillna: float or None, optional (default `None`)
		Value to use for `nan` values in `coeffs`, `None` skips them.
	log3: bool, optional (default `True`)
		Interpolate the coefficients as log(ens)-log(zm)-log(coeff)
		instead of a linear variant.
	rbf: str or callable, optional (default "multiquadric")
		Radial basis functions to use for :class:`scipy.interpolate.Rbf`.

	Returns
	-------
	a_br: array_like (M, N)
		A scalar (0-D) `energy` is promoted to 1-D, and the result will
		have shape (1, N), *not* (N,).
		Energy dissipation rate, units: [keV cm⁻³ s⁻¹]

	See also
	--------
	scipy.interpolate.Rbf
	"""
	energy = np.atleast_1d(np.asarray(energy, dtype=float))
	ens = np.asarray(ens)
	zm_p_en = np.asarray(zm_p_en)

	coeffs = np.array(coeffs, copy=fillna is not None)
	nans = np.isnan(coeffs)
	if fillna is not None:
		coeffs[nans] = fillna
		# update nan positions (should be all False)
		nans = np.isnan(coeffs)

	z = scale_height * rho / energy
	# reshape by `numpy`'s automatic broacdasting to the same shape as z
	enp = np.ones_like(z) * energy

	pts = [
		(_e, _z, coeffs[_i, _j])
		for _i, _e in enumerate(ens)
		for _j, _z in enumerate(zm_p_en)
		if not nans[_i, _j]
	]
	pts = np.array(pts, copy=False)

	if log3:
		enp = np.log(enp)
		z = np.log(z)
		pts = np.log(pts)
	intp = interpolate.Rbf(*(pts.T), function=rbf)
	abr_zm = intp(enp, z)

	if log3:
		abr_zm = np.exp(abr_zm)
	return abr_zm * rho * flux
