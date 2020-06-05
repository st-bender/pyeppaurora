# Copyright (c) 2020 Stefan Bender
#
# This file is part of electronaurora.
# electronaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Atmospheric ionization from auroral electron precipitation

Bundles some of the parametrizations for middle and upper atmospheric
ionization and recombination rates for precipitating
auroral (100 eV--30 keV) and radiation-belt (30 keV--1 MeV) electrons.
"""
__version__ = "0.0.1.dev0"

from .ionization import *
from .ssusi import *
from .recombination import *
