# coding: utf-8
# Copyright (c) 2020 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Particle precipitation spectra

"""

import numpy as np

__all__ = [
	"maxwell_general",
	"pflux_maxwell",
]


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
