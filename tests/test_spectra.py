# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.integrate import quad

import eppaurora.spectra as spec
from eppaurora.electrons import fang2010_mono

# unit particle flux
PFLUX_NNORM = [
	spec.exp_general,
	spec.gaussian_general,
	spec.maxwell_general,
	spec.pow_general,
]

# unit energy flux
PFLUX_ENORM = [
	spec.pflux_exp,
	spec.pflux_gaussian,
	spec.pflux_maxwell,
	spec.pflux_pow,
]


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_NNORM,
)
def test_nflux_norm(pflux_func):
	norm = quad(pflux_func, 0., np.inf)[0]
	np.testing.assert_allclose(norm, 1., rtol=1e-9)
	return


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_ENORM,
)
def test_pflux_norm(pflux_func):
	norm = quad(lambda x: x * pflux_func(x), 0., np.inf)[0]
	np.testing.assert_allclose(norm, 1., rtol=1e-9)
	return


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_ENORM,
)
def test_ediss_spec_int(pflux_func):
	energies = np.logspace(-2, 4, 257)
	dfluxes = pflux_func(energies)
	ediss = spec.ediss_spec_int(
		energies, dfluxes,
		6e5, 5e-10,
		fang2010_mono,
	)
	assert ediss.shape == (1, 1)
	return


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_ENORM,
)
def test_ediss_specfun_int(pflux_func):
	energies = np.logspace(-1, 2, 4)
	fluxes = 1
	# ca. 100 km
	scale_height = 6e5
	rho = 5e-10
	ediss = spec.ediss_specfun_int(
		energies, fluxes,
		scale_height, rho,
		fang2010_mono,
		spec_fun=pflux_func,
	)
	assert ediss.shape == (1, 4)
	return
