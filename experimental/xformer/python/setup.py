#!/usr/bin/python3

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import platform
from setuptools import setup, find_packages
import pathlib
import os

# Find path to xcore-opt binary
here = pathlib.Path(__file__).parent.resolve()
exe_suffix = ".exe" if platform.system() == "Windows" else ""
XCOREOPT_BINARY = pathlib.Path.joinpath(here.parent, "bazel-bin", "xcore-opt",
                                        exe_suffix)

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Get tag version from env variable
# This will be in the format, vX.Y.Z
# We need to remove the first character to get just the version number
environment_variable_name = 'XMOS_AI_TOOLS_RELEASE_VERSION'
VERSION_NUMBER = os.environ.get( environment_variable_name, "0.1.0" )
VERSION_NUMBER = VERSION_NUMBER[1:]

# Force platform specific wheel.
# https://stackoverflow.com/questions/45150304
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            # We don't contain any python extensions so are version agnostic
            # but still want to be platform specific.
            python, abi = 'py3', 'none'
            return python, abi, plat

except ImportError:
    bdist_wheel = None

setup(
    name="xmos-tools",
    version=VERSION_NUMBER,
    author="XMOS",
    author_email="support@xmos.com",
    license="LICENSE.txt",
    description="XMOS AI Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xmos/ai_tools",
    classifiers=[
        "License :: Other/Proprietary License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    data_files=[('bin', [str(XCOREOPT_BINARY)])],
    cmdclass={
        'bdist_wheel': bdist_wheel,
    },
    keywords="tensorflow binarized neural networks",
)
