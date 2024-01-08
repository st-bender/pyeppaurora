# coding: utf-8
# Copyright (c) 2023 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Empirical models for electron energy and flux and ionization rates

Implements the empirical proxy-driven models for auroral electrons,
providing the proxy driven ionization rate model described in [1]_
and the empirical electron energy and flux model described in [2]_.

.. [1] Bender et al., in prep., 2023
.. [2] Zhang and Paxton, JASTP, 70, 1231--1242, 2008, https://doi.org/10.1016/j.jastp.2008.03.008
"""

from .ssusiq2023 import *
from .zhangpaxton2008 import *
