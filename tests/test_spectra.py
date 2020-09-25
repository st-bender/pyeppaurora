# -*- coding: utf-8 -*-
import numpy as np
import pytest

import eppaurora.spectra as spec

# unit particle flux
PFLUX_NNORM = [
	spec.exp_general,
	spec.gaussian_general,
	spec.maxwell_general,
]

# unit energy flux
PFLUX_ENORM = [
	spec.pflux_exp,
	spec.pflux_gaussian,
	spec.pflux_maxwell,
]


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_NNORM,
)
def test_nflux_norm(pflux_func):
	energies = np.logspace(-2, 4, 257)
	spec = pflux_func(energies)
	norm = np.trapz(spec, energies)
	np.testing.assert_allclose(norm, 1., rtol=1e-3)
	return


@pytest.mark.parametrize(
	"pflux_func",
	PFLUX_ENORM,
)
def test_pflux_norm(pflux_func):
	energies = np.logspace(-2, 4, 257)
	spec = pflux_func(energies)
	norm = np.trapz(spec * energies, energies)
	np.testing.assert_allclose(norm, 1., rtol=1e-3)
	return
