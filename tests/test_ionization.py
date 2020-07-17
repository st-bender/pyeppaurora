# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import pytest

import eppaurora as aur


@pytest.mark.parametrize(
	"edissfunc",
	[
		aur.rr1987,
		aur.rr1987_mod,
		aur.fang2008,
		aur.fang2010_mono,
		aur.fang2010_maxw_int,
		aur.fang2013_protons,
		aur.berger1974,
	]
)
def test_endiss(edissfunc):
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
