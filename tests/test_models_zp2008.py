# -*- coding: utf-8 -*-
import numpy as np
import pytest

aurmod = pytest.importorskip(
	"eppaurora.models", reason="Could not import the `models` submodule."
)


@pytest.mark.parametrize(
	"Kp, hp", [
		(2.0, 23.304548167783693),
		(5.0, 69.37903754706306),
		(5.01, 69.6037739596099),
		(7.0, 146.4364572258345),
	]
)
def test_hp(Kp, hp):
	hpp = aurmod.hemispheric_power(Kp)
	np.testing.assert_allclose(hpp, hp)


def test_zp2008():
	q, e = aurmod.zp2008(
		65.0, 23.0, 4.0,
	)
	np.testing.assert_allclose(q, 5.061683414221345)
	np.testing.assert_allclose(e, 5.914083679736101)
