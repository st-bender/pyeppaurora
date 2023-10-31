# -*- coding: utf-8 -*-
import numpy as np
import pytest

xr = pytest.importorskip(
	"xarray",
	reason="`xarray` is needed, it should be installed with the `models` extra."
)
aurmod = pytest.importorskip(
	"eppaurora.models", reason="Could not import the `models` submodule."
)

COEFF_DS = xr.Dataset(
	data_vars={
		"beta": (
			["altitude", "latitude", "mlt", "proxy"],
			np.tile(
				[[[[1., 2., 3., 4., 5., 6.]]]],
				(1, 2, 2, 1),
			),
		),
		"beta_std": (
			["altitude", "latitude", "mlt", "proxy"],
			np.tile(
				[[[[1., 2., 3., 4., 5., 6.]]]],
				(1, 2, 2, 1),
			),
		),
	},
	coords={
		"altitude": [100.],
		"latitude": [70.2, 73.8],
		"mlt": [3.0, 5.0],
		"proxy": ["Kp", "PC", "Ap", "log_f107_81ctr_obs", "log_v_plasma", "offset"],
	},
)


def test_ssusiq2023():
	res = aurmod.ssusiq2023(
		70.2, 3, 100., [2.333, 1, 20, 2, 2], coeff_ds=COEFF_DS,
	)
	np.testing.assert_allclose(res, [2.333 + 2. + 60. + 8. + 10. + 6.])
	res = aurmod.ssusiq2023(
		70.2, 3, 100., [2.333, 1, 20, 2, 2], coeff_ds=COEFF_DS, return_var=True,
	)
	np.testing.assert_allclose(
		res[1], [2.333**2 + 2.**2 + 60.**2 + 8.**2 + 10.**2 + 6.**2]
	)
	res = aurmod.ssusiq2023(
		70.2, 3, 100.,
		[
			[2.333, 1, 20, 2, 2],
			[3.333, 2, 50, 4, 3]
		],
		coeff_ds=COEFF_DS,
		return_var=True,
	)
	np.testing.assert_allclose(
		res[0],
		np.array([
			2.333 + 2. + 60. + 8. + 10. + 6.,
			3.333 + 4. + 150. + 16. + 15. + 6.,
		])
	)
	np.testing.assert_allclose(
		res[1],
		np.array([
			2.333**2 + 2.**2 + 60.**2 + 8.**2 + 10.**2 + 6.**2,
			3.333**2 + 4.**2 + 150.**2 + 16.**2 + 15.**2 + 6.**2,
		])
	)
	res = aurmod.ssusiq2023(
		70.2, 3, 100.,
		[
			[2.333, 3.333], [1, 2], [20, 50], [2, 4], [2, 3]
		],
		coeff_ds=COEFF_DS,
		return_var=True,
	)
	np.testing.assert_allclose(
		res[0],
		np.array([
			2.333 + 2. + 60. + 8. + 10. + 6.,
			3.333 + 4. + 150. + 16. + 15. + 6.,
		])
	)


def test_ssusiq2023_xrda_2d():
	res = aurmod.ssusiq2023(
		70.2, 3, 100.,
		xr.DataArray(
			[[2.333, 3.333], [1, 2], [20, 50], [2, 4], [2, 3]],
			dims=["proxy", "time"],
			coords={"proxy": ["Kp", "PC", "Ap", "log_f107_81ctr_obs", "log_v_plasma"]},
		),
		coeff_ds=COEFF_DS,
		return_var=True,
	)
	assert res[0].shape == (2,)
	np.testing.assert_allclose(
		res[0],
		np.array([
			2.333 + 2. + 60. + 8. + 10. + 6.,
			3.333 + 4. + 150. + 16. + 15. + 6.,
		])
	)


@pytest.mark.parametrize(
	"interpolate, method",
	[
		(False, None),
		(True, "linear"),
		pytest.param(
			True, "cubic",
			marks=pytest.mark.skipif(
				tuple(map(int, xr.__version__.split("."))) < (0, 16),
				reason="'cubic' not supported for N-D interpolation.",
			),
		),
	],
)
def test_ssusiq2023_xrda_3d(interpolate, method):
	res = aurmod.ssusiq2023(
		70.2, 3, 100.,
		xr.DataArray(
			[[[2.333, 3.333], [1, 2], [20, 50], [2, 4], [2, 3]]],
			dims=["model", "proxy", "time"],
			coords={"proxy": ["Kp", "PC", "Ap", "log_f107_81ctr_obs", "log_v_plasma"]},
		),
		interpolate=interpolate,
		method=method,
		return_var=True,
	)
	assert res[0].shape == (1, 2)


@pytest.mark.parametrize(
	"interpolate, method",
	[
		(False, None),
		(True, "linear"),
		pytest.param(
			True, "cubic",
			marks=pytest.mark.skipif(
				tuple(map(int, xr.__version__.split("."))) < (0, 16),
				reason="'cubic' not supported for N-D interpolation.",
			),
		),
	],
)
def test_ssusiq2023_vec(interpolate, method):
	res = aurmod.ssusiq2023(
		[66.6, 70.2], [3, 5, 7], [100., 105., 110., 115.],
		[[2.333], [1], [20], [2]],
		interpolate=interpolate,
		method=method,
		return_var=True,
	)
	assert res[0].shape == (3, 2, 4, 1)
