__author__ = 'Song'
__copyright__ = "Copyright 2016, The Helios Project"

import os.path
import sys, getopt, re
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import pip

global include_dirs
include_dirs = []

deps_list = ['numpy', 'scipy', 'pandas']

def deps_install():
    for package in deps_list:
        print("[DEPENDENCY] Installing %s" % package)
        try:
            pip.main(['install', '--no-binary', ':all:', '--upgrade', package])
        except Exception as e:
            print("[Error] Unable to install %s using pip. \
                  Please read the instructions for \
                  manual installation.. Exiting" % package)
            exit(2)

class pyten_install(install):
    def run(self):
        deps_install()
        import numpy as np
        include_dirs.append(np.get_include())
        install.run(self)

class pyten_develop(develop):
    def run(self):
        deps_install()
        import numpy as np
        include_dirs.append(np.get_include())
        develop.run(self)

local_path = os.path.split(os.path.realpath(__file__))[0]
version_file = os.path.join(local_path, 'pyten/_version.py')
version_strline = open(version_file).read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, version_strline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))

setup(name = "pyten",
      version = version,
      packages=find_packages(),
      include_package_data=True,
      url="",
      author = "HELIOS",
      author_email = "qqsong@tamu.edu",
      description="Tools for the decomposition & completion of tensors",
      #      long_description=open("README.rst").read(),d
      zip_safe = False,         # I need this for MPI purposes
      cmdclass={'install': pyten_install,
                'develop': pyten_develop},
      include_dirs=include_dirs, requires=['numpy', 'scipy', 'pandas']
      )
