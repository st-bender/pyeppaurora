# coding: utf-8
# Copyright (c) 2023 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Empirical model for electron energy and flux

Implements the empirical Kp-driven model for auroral electrons,
providing mean energy and energy flux as described in [1]_.

.. [1] Zhang and Paxton, JASTP, 70, 1231--1242, 2008, https://doi.org/10.1016/j.jastp.2008.03.008
"""
from os import path
from pkg_resources import resource_filename

import numpy as np

__all__ = [
	"epstein_coeffs",
	"epstein_eval",
	"hemispheric_power",
	"read_zp2008_coeffs",
	"zp2008",
]

COEFF_FILE = "Zhang2008.txt"
COEFF_PATH = resource_filename(__name__, path.join("data", COEFF_FILE))
COEFF_NAMES = ["A", "B", "C", "D"]

# Kp_model bin edges and centres as in Zhang and Paxton, 2008
KP_BINE = [
	(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 8.0), (8.0, 10.0)
]
KP_BINC = np.asarray(KP_BINE).mean(axis=1)


def hemispheric_power(Kp):
	"""Hemispheric Power in GW from Kp

	Zhang and Paxton, 2008, Eqs. (1) and (2)

	Parameters
	-----------
	Kp: float or array_like
		Geomagnetic Kp index value(s).

	Returns
	-------
	HP: float or array_like
		Hemispheric power in [GW], same shape as `Kp`.
	"""
	Kp = np.asarray(Kp)
	return np.where(
		Kp <= 5.0,
		38.66 * np.exp(0.1967 * Kp) - 33.99,
		4.592 * np.exp(0.4731 * Kp) + 20.47,
	)


def read_zp2008_coeffs(file=None, nf=6, nKp=len(KP_BINC)):
	"""Read Epstein coefficient tables from file

	Parameters
	----------
	file: str, optional
		The text file containing the coefficient tables, with the same layout
		as in [ZP08]_. Defaults to the packaged coefficient file.
	nf: int, optional
		The number of harmonic (Fourier) terms, defaults to 6 as in [ZP08]_
	nKp: int, optional
		The number of Kp bins, defaults to 6 as in [ZP08]_

	Returns
	-------
	Q0_table, Em_table: tuple of ``numpy.recarray``
		The tables for the total energy flux Q0 and the electron mean energy Em,
		the row names indicate the frequency and the columns the Epstein
		parameters A, B, C, D.

	References
	----------
	.. [ZP08] Zhang and Paxton, JASTP, 70, 1231--1242, 2008,
		https://doi.org/10.1016/j.jastp.2008.03.008
	"""
	fdata = np.genfromtxt(
		file or COEFF_PATH,
		delimiter=" ",
		dtype=None,
		encoding="utf-8",
		names=["name"] + COEFF_NAMES,
	)
	# number of coefficients per Kp bin
	nc = 2 * nf + 1
	# Energy fluxes are in Table 1.
	Q0tab = [fdata[n * nc:(n + 1) * nc] for n in range(nKp)]
	# Mean energies are in Table 2.
	Emtab = [fdata[n * nc:(n + 1) * nc] for n in range(nKp, 2 * nKp)]
	return Q0tab, Emtab


def find_Kp_idx(Kp):
	# maximum avoids negative indices
	return np.maximum(0, np.searchsorted(KP_BINC, Kp, side="left") - 1)


def epstein_coeffs(angle, table):
	r"""Epstein coefficients from table

	Returns the Epstein coefficients as read from the table,
	ready for evaluation with :func:`epstein_eval()`.

	Parameters
	----------
	angle: float or array_like
		The magnetic local time hour angle: :math:`angle = \text{MLT} * 2\pi / 24`
	table: array_like
		Table of N harmonic (Fourier) coefficients.
		The 4 constant offsets are in the first row, then Nx4 cosine amplitudes
		followed by Nx4 sine amplitudes, each for the coefficients A, B, C, and D.

	Returns
	-------
	coeffs: array_like (4,)
		The Epstein coefficients for the MLT angle.

	See Also
	--------
	epstein_eval
	"""
	coeffs = np.array(table[COEFF_NAMES].tolist())
	nf = (len(coeffs) - 1) // 2
	if (2 * nf + 1) != len(coeffs):
		raise ValueError("Number of coefficients is inconsistent.")
	fs = np.arange(1, nf + 1)
	cos = np.cos(fs * angle).dot(coeffs[1:nf + 1])
	sin = np.sin(fs * angle).dot(coeffs[nf + 1:2 * nf + 1])
	return coeffs[0] + cos + sin


def epstein_eval(x, coeffs):
	r"""Epstein function evaluated at x

	The so-called Epstein function is defined by

	.. math:: E(x) = \frac{A\exp\{(x - B) / C\}}{(1 + \exp\{(x - B) / D\})^2}

	Parameters
	----------
	x: float or array_like
		Argument of the Epstein function, e.g. the magnetic co-latitude
		:math:`x = 90 - |Mlat|`.
	coeffs: array_like
		Epstein coefficients, e.g. from :func:`epstein_coeffs()`.

	Returns
	-------
	y: float or array_like
		The Epstein function with coefficients as given evaluated at x.
	"""
	a, b, c, d = coeffs
	loc = x - b
	return a * np.exp(loc / c) / (1 + np.exp(loc / d))**2


def zp2008(mlat, mlt, Kp, Q0table=None, Emtable=None):
	u"""
	Parameters
	----------
	mlat: float
		(Geo)Magnetic latitude in [degrees].
	mlt: float
		Magnetic local time in [hours].
	Kp: float
		Geomagnetic Kp index value(s).
	Q0table: np.recarray, optional
		Fourier coefficient table for the Epstein coefficients
		for the energy flux. E.g. as returned by `read_zp2008_coeffs()`.
	Emtable: np.recarray, optional
		Fourier coefficient table for the Epstein coefficients
		for the mean energy. E.g. as returned by `read_zp2008_coeffs()`.

	Returns
	-------
	(Q0, Em): tuple
		Electron energy flux Q0 in [mW m⁻²] (or [erg s⁻¹ cm⁻²]),
		and electron mean energy in [keV].
	"""
	if (Q0table is None) or (Emtable is None):
		Q0t, Emt = read_zp2008_coeffs()
		Q0table = Q0table or Q0t
		Emtable = Emtable or Emt

	angle = mlt * np.pi / 12.0
	x = 90.0 - np.abs(mlat)
	ix = find_Kp_idx(Kp)
	ixs = [ix, ix + 1]
	Kps = KP_BINC[ixs]
	Q0_epst = [
		epstein_eval(x, epstein_coeffs(angle, Q0table[ix]))
		for ix in ixs
	]
	Em_epst = [
		epstein_eval(x, epstein_coeffs(angle, Emtable[ix]))
		for ix in ixs
	]
	Q0 = np.interp(hemispheric_power(Kp), hemispheric_power(Kps), Q0_epst)
	Em = np.interp(Kp, Kps, Em_epst)
	return Q0, Em
