# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import pytest

import eppaurora as aur

EDISS_FUNCS_EXPECTED = [
	(aur.rr1987, 4.51517584e-07),
	(aur.rr1987_mod, 4.75296602e-07),
	(aur.fang2008, 4.44256875e-07),
	(aur.fang2010_mono, 1.96516057e-007),
	(aur.fang2010_maxw_int, 4.41340659e-07),
	(aur.fang2013_protons, 4.09444686e-22),
	(aur.berger1974, 2.37365609e-03),
]


@pytest.mark.parametrize(
	"edissfunc, expected",
	EDISS_FUNCS_EXPECTED,
)
def test_endiss(edissfunc, expected):
	energies = np.logspace(-1, 2, 4)
	fluxes = np.ones_like(energies)
	# ca. 100, 150, 200 km
	scale_heights = np.array([6e5, 27e5, 40e5])
	rhos = np.array([5e-10, 1.7e-12, 2.6e-13])
	# energy dissipation "profiles"
	ediss = edissfunc(
		energies[None, :], fluxes[None, :],
		scale_heights[:, None], rhos[:, None]
	)
	assert ediss.shape == (3, 4)
	np.testing.assert_allclose(ediss[0, 2], expected)
	return


def test_ssusi_ioniz():
	energies = np.logspace(-1, 2, 4)
	fluxes = np.ones_like(energies)
	z = np.array([100, 120, 150])
	# energy dissipation "profiles"
	ediss = aur.ssusi_ioniz(
		z[:, None],
		energies[None, :], fluxes[None, :],
	)
	assert ediss.shape == (3, 4)
	return
