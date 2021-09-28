# coding: utf-8
# Copyright (c) 2021 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric conductivity from electron densities [1]_ [2]_ [3]_

.. [1] A. Brekke, J. R. Doupnik, P. M. Banks,
	Incoherent scatter measurements of E region conductivities and currents in the auroral zone,
	J. Geophys. Res., 79(25), 3773--3790, Sept. 1974,
	doi: `10.1029/JA079i025p03773 <https://doi.org/10.1029/JA079i025p03773>`_

.. [2] James F. Vickrey, Richard R. Vondrak, Stephen J. Matthews,
	The diurnal and latitudinal variation of auroral zone ionospheric conductivity,
	J. Geophys. Res., 86(A1), 65--75, Jan. 1981,
	doi: `10.1029/JA086iA01p00065 <https://doi.org/10.1029/JA086iA01p00065>`_

.. [3] R. M. Robinson, R. R. Vondrak, K. Miller, T. Dabbs, D. Hardy,
	On calculating ionospheric conductances from the flux and energy of precipitating electrons,
	J. Geophys. Res. Space Phys., 92(A3), 2565--2569, Mar. 1987,
	doi: `10.1029/JA092iA03p02565 <https://doi.org/10.1029/JA092iA03p02565>`_
"""

import numpy as np

__all__ = [
	"ion_coll",
	"ion_gyro",
	"pedersen",
	"hall",
	"SigmaP_robinson1987",
	"SigmaH_robinson1987",
]

E_CHARGE = 1.602176634e-19  # [C] = [As]


def ion_coll(n_neutral):
	"""Ion--neutral collision frequency

	Derived from the atmospheric neutral density [#]_
	e.g., from NRLMSISE-00.

	Parameters
	----------
	n_neutral: float or array_like (M, ...)
		Atmospheric density in [cm⁻³]

	Returns
	-------
	ion_coll: float or array_like (M, ...)
		Ion--neutral collision frequency [s⁻¹]

	References
	----------
	.. [#] Vickrey et al., J. Geophys. Res., 86(A1), 65--75, Jan. 1981,
		doi: `10.1029/JA086iA01p00065 <https://doi.org/10.1029/JA086iA01p00065>`_
	"""
	return 3.75e-10 * n_neutral


def ion_gyro(bmag, m_ion=30.):
	"""Ion gyro (cyclotron) frequency

	Parameters
	----------
	bmag: float or array_like (M,...)
		Magnitude of the magnetic B-field [T].
	m_ion: float or array_like (M, ...), optional (default 30.0)
		Ion mass [GeV / c²]

	Returns
	-------
	ion_gyro: float or array_like (M, ...)
		Ion cyclotron frequency [s⁻¹]
	"""
	return 2 * np.pi * 2.8e10 * bmag * (511e-6 / m_ion)


def pedersen(ne, bmag, ion_gyro, ion_coll):
	"""Pedersen conductivity σP

	Formulae and parameters as described in [#]_ [#]_,
	neglecting electron--neutral collisions.

	Parameters
	----------
	ne: float or array_like (M, ...)
		Electron density in [m⁻³]
	bmag: float or array_like (M, ...)
		Magnitude of the magnetic B-field [T].
	ion_gyro: float or array_like (N, ...)
		The ion gyro frequency [s⁻¹]
	ion_coll: float or array_like (N, ...)
		The ion collision frequency [s⁻¹]

	Returns
	-------
	σP: float or array_like (M, N) if broadcastable
		Pedersen conductivity [S m⁻¹].

	References
	----------
	.. [#] Brekke et al., J. Geophys. Res., 79(25), 3773--3790, Sept. 1974,
		doi: `10.1029/JA079i025p03773 <https://doi.org/10.1029/JA079i025p03773>`_

	.. [#] Vickrey et al., J. Geophys. Res., 86(A1), 65--75, Jan. 1981,
		doi: `10.1029/JA086iA01p00065 <https://doi.org/10.1029/JA086iA01p00065>`_
	"""
	return ne * E_CHARGE / bmag * ion_gyro * ion_coll / (ion_gyro**2 + ion_coll**2)


def hall(ne, bmag, ion_gyro, ion_coll):
	"""Hall conductivity σH

	Formulae and parameters as described in [#]_ [#]_,
	neglecting electron--neutral collisions.

	Parameters
	----------
	ne: float or array_like (M, ...)
		Electron density in [m⁻³]
	bmag: float or array_like (M,...)
		Magnitude of the magnetic B-field [T].
	ion_gyro: float or array_like (N, ...)
		The ion gyro frequency [s⁻¹]
	ion_coll: float or array_like (N, ...)
		The ion collision frequency [s⁻¹]

	Returns
	-------
	σH: float or array_like (M, N) if broadcastable
		Hall conductivity [S m⁻¹].

	References
	----------
	.. [#] Brekke et al., J. Geophys. Res., 79(25), 3773--3790, Sept. 1974,
		doi: `10.1029/JA079i025p03773 <https://doi.org/10.1029/JA079i025p03773>`_

	.. [#] Vickrey et al., J. Geophys. Res., 86(A1), 65--75, Jan. 1981,
		doi: `10.1029/JA086iA01p00065 <https://doi.org/10.1029/JA086iA01p00065>`_
	"""
	return ne * E_CHARGE / bmag * ion_coll**2 / (ion_gyro**2 + ion_coll**2)


def SigmaP_robinson1987(en_avg, flx):
	"""Pedersen conductance [#]_

	Directly derived from the electron mean energy and energy flux.

	Parameters
	----------
	en_avg: float or array_like (M, ...)
		Electron average energy in [keV]
	flx: float or array_like (M, ...)
		Energy flux [ergs cm⁻² s⁻¹].

	Returns
	-------
	ΣP: float or array_like (M, N) if broadcastable
		Pedersen conductance [S].

	References
	----------
	.. [#] Robinson et al., J. Geophys. Res. Space Phys., 92(A3), 2565--2569, Mar. 1987,
		doi: `10.1029/JA092iA03p02565 <https://doi.org/10.1029/JA092iA03p02565>`_
	"""
	return 40 * en_avg / (16. + en_avg**2) * np.sqrt(flx)


def SigmaH_robinson1987(en_avg, flx):
	"""Hall conductance [#]_

	Directly derived from the electron mean energy and energy flux.

	Parameters
	----------
	en_avg: float or array_like (M, ...)
		Electron average energy in [keV]
	flx: float or array_like (M, ...)
		Energy flux [ergs cm⁻² s⁻¹].

	Returns
	-------
	ΣH: float or array_like (M, N) if broadcastable
		Hall conductance [S].

	References
	----------
	.. [#] Robinson et al., J. Geophys. Res. Space Phys., 92(A3), 2565--2569, Mar. 1987,
		doi: `10.1029/JA092iA03p02565 <https://doi.org/10.1029/JA092iA03p02565>`_
	"""
	return 0.45 * en_avg**(0.85) * SigmaP_robinson1987(en_avg, flx)
