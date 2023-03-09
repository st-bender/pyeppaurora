# coding: utf-8
# Copyright (c) 2023 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Empirical model for auroral ionization rates

Implements the empirical model for auroral ionization,
derived from SSUSI UV observations [1]_.

.. [1] Bender et al., in prep., 2023
"""
from os import path
from pkg_resources import resource_filename

import numpy as np
import xarray as xr

__all__ = [
	"ssusiq2023",
]

COEFF_FILE = "SSUSI_IRgrid_coeffs_f17f18.nc"
COEFF_PATH = resource_filename(__name__, path.join("data", COEFF_FILE))


def ssusiq2023(gmlat, mlt, alt, sw_coeffs, coeff_ds=None, return_var=False):
	u"""
	Parameters
	----------
	gmlat: float
		Geomagnetic latitude in [degrees].
	mlt: float
		Magnetic local time in [hours].
	alt: float
		Altitude in [km]
	sw_coeffs: array_like or `xarray.DataArray`
		The space weather index values to use (for the requested time(s)),
		should be of shape (N, M) with N = number of proxies, currently 5:
		[Kp, PC, Ap, log(f10.7_81ctr_obs), log(v_plasma)].
		The `xarray.DataArray` should have a dimension named "proxy" with
		matching coordinates:
		["Kp", "PC", "Ap", "log_f107_81ctr_obs", "log_v_plasma"]
		All the other dimensions will be broadcasted.
	coeff_ds: `xarray.Dataset`, optional (default: None)
		Dataset with the model coefficients, `None` uses the packaged version.
	return_var: bool, optional (default: False)
		If `True`, returns the predicted variance in addition to the values,
		otherwise only the mean prediction is returned.

	Returns
	-------
	q: `xarray.DataArray`
		log(q), where q is the ionization rate in [cm⁻³ s⁻¹]
		if `return_var` is False.
	q, var(q): tuple of `xarray.DataArray`s
		log(q) and var(log(q)) where q is the ionization rate in [cm⁻³ s⁻¹]
		if `return_var` is True.
	"""
	coeff_ds = coeff_ds or xr.open_dataset(
		COEFF_PATH, decode_times=False, engine="h5netcdf"
	)
	coeff_sel = coeff_ds.sel(
		altitude=alt, latitude=gmlat, mlt=mlt, method="nearest",
	)

	# Determine if `xarray` read bytes or strings to
	# match the correct name in the proxy names.
	# Default is plain strings.
	offset = "offset"
	if isinstance(coeff_ds.proxy.values[0], bytes):
		offset = offset.encode()
	have_offset = offset in coeff_ds.proxy.values

	# prepare the coefficients (array) as a `xarray.DataArray`
	if isinstance(sw_coeffs, xr.DataArray):
		if have_offset:
			ones = xr.ones_like(sw_coeffs.isel(proxy=0))
			ones = ones.assign_coords(proxy="offset")
			sw_coeffs = xr.concat([sw_coeffs, ones], dim="proxy")
	else:
		sw_coeffs = np.atleast_2d(sw_coeffs)
		if have_offset:
			aix = sw_coeffs.shape.index(len(coeff_ds.proxy.values) - 1)
			if aix != 0:
				sw_coeffs = sw_coeffs.T
			sw_coeffs = np.vstack([sw_coeffs, np.ones(sw_coeffs.shape[1])])
		else:
			aix = sw_coeffs.shape.index(len(coeff_ds.proxy.values))
			if aix != 0:
				sw_coeffs = sw_coeffs.T
		extra_dims = ["dim_{0}".format(_d) for _d in range(sw_coeffs.ndim - 1)]
		sw_coeffs = xr.DataArray(
			sw_coeffs,
			dims=["proxy"] + extra_dims,
			coords={"proxy": coeff_ds.proxy.values},
		)

	# Calculate model (mean) values from `beta`
	# fill NaNs with zero for `.dot()`
	coeffs = coeff_sel.beta.fillna(0.)
	q = coeffs.dot(sw_coeffs)
	if not return_var:
		return q

	# Calculate variance of the model from `beta_std`
	# fill NaNs with zero for `.dot()`
	coeffv = coeff_sel.beta_std.fillna(0.)**2
	q_var = coeffv.dot(sw_coeffs**2)
	return q, q_var
