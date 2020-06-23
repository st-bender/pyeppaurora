# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric ionization rate parametrizations

From the SSUSI ATBD documents.

.. [#] https://ssusi.jhuapl.edu/data_algorithms
.. [#] https://ssusi.jhuapl.edu/docs/algorithms/Aurora_LID_c_Version_2.0.pdf
.. [#] https://ssusi.jhuapl.edu/docs/algorithms/SSUSI_DataProductAlgorithms_V1_13.doc
"""

import numpy as np

__all__ = ["ssusi_ioniz"]

# pre-determined analytical model coefficients of peak auroral ionization production rate height
# electrons
CHMAX_E = [2.07923, -9.41205e-2]
# protons
CHMAX_P = [2.078, -4.072e-2]
# pre-determined analytical model coefficients of peak auroral ionization production rate
# electrons
CPMAX_E = [0., 9.25777e-1, -5.03201e-1]
# protons
CPMAX_P = [0., 3.50766e-1, -8.84737e-2]


def ssusi_ioniz(z, en, flux, chmax=CHMAX_E, cpmax=CPMAX_E, eref=1., pref=2.57e3, shpc=1.427e10):
	"""Parametrization from Sect. 2.6.2 in [#]_

	Parameters
	----------
	z: float, array_like
	en: float, array_like
	flux: float, array_like
		Energy flux in [erg cm^{-2} s^{-2}], note: **not** keV.
	chmax: tuple, list, (2,) optional
		Pre-determined analytical model coefficients of peak auroral ionization production rate height.
	cpmax: tuple, list, (3,) optional
		Pre-determined analytical model coefficients of peak auroral ionization production rate
		
	Returns
	-------
	q: float, array_like
		The atmospheric ionization rate at altitude z.

	References
	----------
	
	.. [#] https://ssusi.jhuapl.edu/docs/algorithms/Aurora_LID_c_Version_2.0.pdf
	"""
	# pre-determined scale height proportionality factor
	shpf = 1e-5 / np.exp(1.)
	# log10 ratio of the characteristic energy (Sect. 2.6.2.2, 2.6.2.4)
	l10rce = np.log10(en / eref)
	# peak auroral ionization production rate height (Sect. 2.6.2.6)
	tempX = np.polyval(chmax[::-1], l10rce)
	pprh = 10**tempX
	# peak auroral ionization production rate (Sect. 2.6.2.8)
	tempX = np.polyval(cpmax[::-1], l10rce)
	ppr1 = 10**(tempX) * pref
	# scale height of the auroral ionization production rate (Sect. 2.6.2.10)
	shpr = shpf * shpc / ppr1
	# electron peak auroral ionization production rate (Sect. 2.6.2.12)
	pprq = flux * ppr1
	# ionization production rate altitude profile (Sect. 2.6.2.15)
	rhpr = (z - pprh) / shpr
	q = pprq * np.exp(1. - rhpr - np.exp(-rhpr))
	return q
