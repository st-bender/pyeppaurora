# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric recombination rate parametrizations

Atmospheric recombination rate parametrizations as described
in [1]_, [2]_, and [3]_.

.. [1] Vickrey et al., J. Geophys. Res. Space Phys., 87, A7, 5184--5196,
	doi:10.1029/ja087ia07p05184
.. [2] Gledhill, Radio Sci., 21, 3, 399-408, doi:10.1029/rs021i003p00399
.. [3] https://ssusi.jhuapl.edu/data_algorithms
"""

import numpy as np

__all__ = [
	"alpha_vickrey1982",
	"alpha_gledhill1986_aurora",
	"alpha_gledhill1986_day",
	"alpha_gledhill1986_night",
	"alpha_ssusi",
]


def alpha_vickrey1982(h):
	u""" Vickrey et al. 1982 [1]_

	Parameters
	----------
	h: float or array_like
		Altitude in [km]

	Returns
	-------
	alpha: float or array_like
		The recombination rate [cm³ s⁻¹].

	.. [1] Vickrey et al., J. Geophys. Res. Space Phys.,
		87, A7, 5184--5196, doi:10.1029/ja087ia07p05184
	"""
	return 2.5e-6 * np.exp(-h / 51.2)


def alpha_gledhill1986_aurora(h):
	""" Gledhill 1986, Aurora parameterization [1]_

	Parameters
	----------
	h: float or array_like
		Altitude in [km]

	Returns
	-------
	alpha: float or array_like
		The recombination rate [cm³ s⁻¹].

	.. [1] Radio Sci., 21, 3, 399-408, doi:10.1029/rs021i003p00399
	"""
	return 4.3e-6 * np.exp(-2.42e-2 * h) + 8.16e12 * np.exp(-0.524 * h)


def alpha_gledhill1986_day(h):
	u""" Gledhill 1986, day-time parameterization [1]_

	Parameters
	----------
	h: float or array_like
		Altitude in [km]

	Returns
	-------
	alpha: float or array_like
		The recombination rate [cm³ s⁻¹].

	.. [1] Radio Sci., 21, 3, 399-408, doi:10.1029/rs021i003p00399
	"""
	return 0.501 * np.exp(-0.165 * h)


def alpha_gledhill1986_night(h):
	""" Gledhill 1986, night-time parameterization [1]_

	Parameters
	----------
	h: float or array_like
		Altitude in [km]

	Returns
	-------
	alpha: float or array_like
		The recombination rate [cm³ s⁻¹].

	.. [1] Radio Sci., 21, 3, 399-408, doi:10.1029/rs021i003p00399
	"""
	return 652 * np.exp(-0.234 * h)


def alpha_ssusi(z, alpha0=4.2e-7, scaleh=28.9, z0=108., z1=None):
	u"""
	Implements section 2.6.2.15 in [2]_.

	Parameters
	----------
	z: float, array_like
		Profile altitude [km].
	alpha0: float, optional
		Predetermined peak effective recombination coefficient.
		Default: 4.2e-7
	scaleh: float, optional
		Predetermined scale height [km] of the effective recombination coefficient.
		Default: 28.9.
	z0: float, optional
		Predetermined altitude [km] of the peak effective recombination coefficient.
		Default: 108.
	z1: float, optional
		Use :func:`alpha_vickrey1982()` above z1 [km].
		Default: None

	Returns
	-------
	alpha: float or array_like
		The recombination rate [cm³ s⁻¹].

	References
	----------
	.. [1] https://ssusi.jhuapl.edu/data_algorithms
	.. [2] https://ssusi.jhuapl.edu/docs/algorithms/Aurora_LID_c_Version_2.0.pdf
	.. [3] https://ssusi.jhuapl.edu/docs/algorithms/SSUSI_DataProductAlgorithms_V1_13.doc
	"""
	alpha = np.zeros_like(z)
	alpha[z < z0] = alpha0
	alpha[z >= z0] = alpha0 * np.exp(-(z[z >= z0] - z0) / scaleh)
	if z1 is not None:
		# use Vickrey et al. above z1
		alpha[z >= z1] = 2.5e-6 * np.exp(-z[z >= z1] / 51.2)
	return alpha
