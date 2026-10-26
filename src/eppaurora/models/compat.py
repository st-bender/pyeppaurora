# coding: utf-8
# Copyright (c) 2026 Stefan Bender
#
# This file is part of pyeppaurora.
# pyeppaurora is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Compatibility helper for package resource files

https://importlib-resources.readthedocs.io/en/latest/migration.html
"""
from os import path


def resource_filepath(file, subdir="data"):
	try:
		from importlib import resources
		ref = resources.files(__package__) / subdir / file
		with resources.as_file(ref) as filepath:
			return filepath
	except (AttributeError, ImportError):
		from pkg_resources import resource_filename
		filepath = resource_filename(__package__, path.join(subdir, file))
	return filepath
