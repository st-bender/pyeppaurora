# -*- coding: utf-8 -*-
import numpy as np
import pytest

import eppaurora as aur

COND1_FUNCS_EXPECTED = [
	(aur.pedersen, [1.50700987e-05, 9.03908593e-06, 8.54346201e-07]),
	(aur.hall, [4.59146798e-04, 1.05810738e-06, 1.30470569e-08]),
]

COND2_FUNCS_EXPECTED = [
	(aur.SigmaP_robinson1987, [0.24984385, 2.35294118, 3.44827586, 0.39936102]),
	(aur.SigmaH_robinson1987, [0.01588112, 1.05882353, 10.98536562, 9.00695907]),
]


@pytest.mark.parametrize(
	"condfunc, expected",
	COND1_FUNCS_EXPECTED,
)
def test_conductivity(condfunc, expected):
	_eV_J = 1.602176634e-19  # eV in J in new SI units
	_erg_J = 1e-7  # 1 erg = 100 nJ
	_erg_keV = _erg_J / _eV_J * 1e-3
	_bmag = 50000.  # nT
	energies = np.logspace(-1, 2, 4)  # keV
	fluxes = np.ones_like(energies)  # ergs / cm² / s = 100 nW / cm²
	# convert to keV / cm² / s
	fluxes = fluxes * _erg_keV
	# ca. 100, 150, 200 km
	alts = np.array([100, 150, 200])
	scale_heights = np.array([6e5, 27e5, 40e5])
	rhos = np.array([5e-10, 1.7e-12, 2.6e-13])
	nds = np.array([12173395869016.26, 46771522616.48675, 6101757504.939191])
	# ionization rates
	qs = 1 / 0.035 * aur.fang2010_maxw_int(
		energies[None, :], fluxes[None, :],
		scale_heights[:, None], rhos[:, None],
	)
	# electron densities
	nes = np.sqrt(qs / aur.recombination.alpha_gledhill1986_aurora(alts)[:, None])
	# gyro frequency
	omega = aur.conductivity.ion_gyro(_bmag * 1e-9)
	# collision frequency
	nu = aur.conductivity.ion_coll(nds)
	# conductances
	cond = condfunc(nes * 1e6, _bmag * 1e-9, omega, nu[:, None])
	assert cond.shape == (3, 4)
	np.testing.assert_allclose(cond[:, 2], expected)
	return


@pytest.mark.parametrize(
	"condfunc, expected",
	COND2_FUNCS_EXPECTED,
)
def test_conductance(condfunc, expected):
	energies = np.logspace(-1, 2, 4)  # keV
	fluxes = np.ones_like(energies)  # ergs / cm² / s
	# conductances
	cond = condfunc(energies, fluxes)
	assert cond.shape == (4,)
	np.testing.assert_allclose(cond, expected, atol=1e-9)
	return
