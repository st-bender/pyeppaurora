from codecs import open
from os import path
import re
# Always prefer setuptools over distutils
from setuptools import find_packages, setup

name = "eppaurora"
meta_path = path.join("src", name, "__init__.py")
here = path.abspath(path.dirname(__file__))

extras_require = {
	"models": ["h5netcdf", "xarray"],
	"tests": ["pytest"],
}
extras_require["all"] = sorted(
	{v for req in extras_require.values() for v in req}
)


# Approach taken from
# https://packaging.python.org/guides/single-sourcing-package-version/
# and the `attrs` package https://www.attrs.org/
# https://github.com/python-attrs/attrs
def read(*parts):
	"""
	Builds an absolute path from *parts* and and return the contents of the
	resulting file.  Assumes UTF-8 encoding.
	"""
	with open(path.join(here, *parts), "rb", "utf-8") as f:
		return f.read()


def find_meta(meta, *path):
	"""
	Extract __*meta*__ from *path* (can have multiple components)
	"""
	meta_file = read(*path)
	meta_match = re.search(
		r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M,
	)
	if not meta_match:
		raise RuntimeError("__{meta}__ string not found.".format(meta=meta))
	return meta_match.group(1)


# Get the long description from the README file
long_description = read("README.md")
version = find_meta("version", meta_path)

if __name__ == "__main__":
	setup(
		name=name,
		version=version,
		description="Atomspheric ionization from auroral particle precipitation",
		long_description=long_description,
		long_description_content_type="text/markdown",
		url="https://github.com/st-bender/pyeppaurora",
		author="Stefan Bender",
		author_email="stefan.bender@ntnu.no",
		license="GPLv2",
		classifiers=[
			"Development Status :: 4 - Beta",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
			"Programming Language :: Python",
			"Programming Language :: Python :: 2",
			"Programming Language :: Python :: 3",
			'Topic :: Scientific/Engineering :: Physics',
			'Topic :: Utilities',
		],
		packages=find_packages("src"),
		package_dir={"": "src"},
		package_data={
			name + ".models": [
				"models/data/SSUSI_IRgrid_coeffs_f17f18.nc",
			],
		},
		include_package_data=True,
		install_requires=[
			"numpy>=1.13.0",
			"scipy>=0.9",  # LinearNDInterpolator
		],
		extras_require=extras_require,
		scripts=[],
		entry_points={},
		zip_safe=False,
	)
