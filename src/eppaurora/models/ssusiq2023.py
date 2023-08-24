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
from logging import warning as warn
from os import path
from pkg_resources import resource_filename

import numpy as np
import xarray as xr

__all__ = [
	"ssusiq2023",
]

COEFF_FILE = "SSUSI_IRgrid_coeffs_f17f18.nc"
COEFF_PATH = resource_filename(__name__, path.join("data", COEFF_FILE))


def _interp(ds, method="linear", method_non_numeric="nearest", **kwargs):
	"""Fix `xarray` interpolation with non-numeric variables
	"""
	v_n = sorted(
		filter(lambda _v: np.issubdtype(ds[_v].dtype, np.number), ds)
	)
	v_nn = sorted(set(ds) - set(v_n))
	ds_n = ds[v_n].interp(method=method, **kwargs)
	ds_nn = ds[v_nn].sel(method=method_non_numeric, **kwargs)
	# override coordinates for `merge()`
	ds_nn = ds_nn.assign_coords(**ds_n.coords)
	return xr.merge([ds_n, ds_nn], join="left")


def ssusiq2023(
	gmlat,
	mlt,
	alt,
	sw_coeffs,
	coeff_ds=None,
	interpolate=False,
	method="linear",
	return_var=False,
):
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
	interpolate: bool, optional (default: False)
		If `True`, uses bilinear interpolate in MLT and geomagnetic latitude,
		using periodic (24h) boundary conditions in MLT. Otherwise, the closest
		MLT/geomagnetic latitude bin will be selected.
	method: str, optional (default: "linear")
		Interpolation method to use, see `scipy.interpolate.interpn` for options.
		Only used if `interpolate` is `True`.
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
	coeff_sel = coeff_ds.sel(altitude=alt)
	if interpolate:
		_ds_m = coeff_sel.assign_coords(mlt=coeff_sel.mlt - 24)
		_ds_p = coeff_sel.assign_coords(mlt=coeff_sel.mlt + 24)
		_ds_mp = xr.concat([_ds_m, coeff_sel, _ds_p], dim="mlt")
		# square the standard deviation for interpolation
		_ds_mp["beta_var"] = _ds_mp["beta_std"]**2
		coeff_sel = _interp(
			_ds_mp,
			latitude=gmlat, mlt=mlt,
			method=method,
		)
		# and square root back to get the standard deviation
		coeff_sel["beta_std"] = np.sqrt(coeff_sel["beta_var"])
	else:
		coeff_sel = coeff_sel.sel(latitude=gmlat, mlt=mlt, method="nearest")

	# Determine if `xarray` read bytes or strings to
	# match the correct name in the proxy names.
	# Default is plain strings.
	offset = "offset"
	if isinstance(coeff_sel.proxy.values[0], bytes):
		offset = offset.encode()
	have_offset = offset in coeff_sel.proxy.values

	# prepare the coefficients (array) as a `xarray.DataArray`
	if isinstance(sw_coeffs, xr.DataArray):
		if have_offset:
			ones = xr.ones_like(sw_coeffs.isel(proxy=0))
			ones = ones.assign_coords(proxy="offset")
			sw_coeffs = xr.concat([sw_coeffs, ones], dim="proxy")
		sw_coeffs = sw_coeffs.sel(proxy=coeff_sel.proxy.astype(sw_coeffs.proxy.dtype))
	else:
		sw_coeffs = np.atleast_2d(sw_coeffs)
		if have_offset:
			aix = sw_coeffs.shape.index(len(coeff_sel.proxy.values) - 1)
			if aix != 0:
				warn(
					"Automatically changing axis. "
					"This is ambiguous, to remove the ambiguity, "
					"make sure that the different indexes (proxies) "
					"are ordered along the zero-th axis in multi-"
					"dimensional settings. I.e. each row corresponds "
					"to a different index, Kp, PC, Ap, etc."
				)
				sw_coeffs = sw_coeffs.swapaxes(aix, 0)
			sw_coeffs = np.vstack([sw_coeffs, np.ones(sw_coeffs.shape[1])])
		else:
			aix = sw_coeffs.shape.index(len(coeff_sel.proxy.values))
			if aix != 0:
				warn(
					"Automatically changing axis. "
					"This is ambiguous, to remove the ambiguity, "
					"make sure that the different indexes (proxies) "
					"are ordered along the zero-th axis in multi-"
					"dimensional settings. I.e. each row corresponds "
					"to a different index, Kp, PC, Ap, etc."
				)
				sw_coeffs = sw_coeffs.swapaxes(aix, 0)
		extra_dims = ["dim_{0}".format(_d) for _d in range(sw_coeffs.ndim - 1)]
		sw_coeffs = xr.DataArray(
			sw_coeffs,
			dims=["proxy"] + extra_dims,
			coords={"proxy": coeff_sel.proxy.values},
		)

	# Calculate model (mean) values from `beta`
	# fill NaNs with zero for `.dot()`
	coeffs = coeff_sel.beta.fillna(0.)
	q = coeffs.dot(sw_coeffs)
	q = q.rename("log_q")
	q.attrs = {
		"long_name": "natural logarithm of ionization rate",
		"units": "log(cm-3 s-1)",
	}
	if not return_var:
		return q

	# Calculate variance of the model from `beta_std`
	# fill NaNs with zero for `.dot()`
	coeffv = coeff_sel.beta_std.fillna(0.)**2
	q_var = coeffv.dot(sw_coeffs**2)
	if "sigma2" in coeff_sel.data_vars:
		# if available, add the posterior variance
		# to get the full posterior predictive variance
		q_var = coeff_sel["sigma2"] + q_var
	q_var = q_var.rename("var_log_q")
	q_var.attrs = {
		"long_name": "variance of the natural logarithm of ionization rate",
		"units": "1",
	}
	return q, q_var
