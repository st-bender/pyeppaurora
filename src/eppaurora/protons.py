# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric ionization rate parametrizations

Includes the atmospheric ionization rate parametrization for auroral
proton precipitation [1]_.

.. [1] Fang, X., Lummerzheim, D., and Jackman, C. H. (2013),
	Proton impact ionization and a fast calculation method,
	J. Geophys. Res. Space Physics, 118, 5369--5378, doi:10.1002/jgra.50484.
"""

import numpy as np
from numpy.polynomial.polynomial import polyval

__all__ = ["fang2013_protons"]

POLY_F2013 = [
	[ 2.55050e+0,  2.69476e-1, -2.58425e-1,  4.43190e-2],
	[ 6.39287e-1, -1.85817e-1, -3.15636e-2,  1.01370e-2],
	[ 1.63996e+0,  2.43580e-1,  4.29873e-2,  3.77803e-2],
	[-2.13479e-1,  1.42464e-1,  1.55840e-2,  1.97407e-3],
	[-1.65764e-1,  3.39654e-1, -9.87971e-3,  4.02411e-3],
	[-3.59358e-2,  2.50330e-2, -3.29365e-2,  5.08057e-3],
	[-6.26528e-1,  1.46865e+0,  2.51853e-1, -4.57132e-2],
	[ 1.01384e+0,  5.94301e-2, -3.27839e-2,  3.42688e-3],
	[-1.29454e-6, -1.43623e-1,  2.82583e-1,  8.29809e-2],
	[-1.18622e-1,  1.79191e-1,  6.49171e-2, -3.99715e-3],
	[ 2.94890e+0, -5.75821e-1,  2.48563e-2,  8.31078e-2],
	[-1.89515e-1,  3.53452e-2,  7.77964e-2, -4.06034e-3]
]


def fang2013_protons(energy, flux, scale_height, rho, pij=POLY_F2013):
	"""Proton ionization parametrization by Fang et al., 2013 [1]_

	.. [1] Fang, X., Lummerzheim, D., and Jackman, C. H. (2013),
		Proton impact ionization and a fast calculation method,
		J. Geophys. Res. Space Physics, 118, 5369--5378, doi:10.1002/jgra.50484.
	"""
	def _f_y(_c, _y):
		# Fang et al., 2008, Eq. (6), Fang et al., 2010 Eq. (4)
		# Fang et al., 2013, Eqs. (6), (7)
		return (
			_c[0] * (_y**_c[1]) * np.exp(-_c[2] * (_y**_c[3])) +
			_c[4] * (_y**_c[5]) * np.exp(-_c[6] * (_y**_c[7])) +
			_c[8] * (_y**_c[9]) * np.exp(-_c[10] * (_y**_c[11]))
		)

	pij = np.asarray(pij)
	# Fang et al., 2013, Eqs. (6), (7)
	_cs = np.exp(polyval(np.log(energy), pij.T))
	# Fang et al., 2013, Eq. (5)
	y = 7.5 / energy * (1e4 * rho * scale_height)**(0.9)
	f_y = _f_y(_cs, y)
	# Fang et al., 2013, Eq. (3)
	en_diss = f_y * flux / scale_height
	return en_diss
