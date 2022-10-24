#!/usr/bin/python3

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import platform
from typing import Union

from setuptools import setup, find_packages
from setuptools.command.install import install
import pathlib

# Find path to xcore-opt binary
here = pathlib.Path(__file__).parent.resolve()
exe_suffix = ".exe" if platform.system() == "Windows" else ""
XCOREOPT_BINARY: Union[pathlib.Path, str] = pathlib.Path.joinpath(
    here.parent, "experimental", "xformer", "bazel-bin", "xcore-opt"
)
XCOREOPT_BINARY = str(XCOREOPT_BINARY) + exe_suffix

# Get the long description from the README file
LONG_README = (here / "README.md").read_text(encoding="utf-8")

# xtflm_interpreter path and libs from lib_tflite_micro
XTFLM_INTERPRETER_LIBS = [
    "libs/linux/xtflm_python.so",
    "libs/linux/xtflm_python.so.1.0.1",
    "libs/macos/xtflm_python.dylib",
    "libs/macos/xtflm_python.1.0.1.dylib",
    "libs/windows/xtflm_python.dll",
]

# xtflm_interpreter requires numpy
REQUIRED_PACKAGES = [
    "numpy<2.0",
    "tflite>=2.4.0",
]

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
            python, abi = "py3", "none"
            return python, abi, plat

except ImportError:
    bdist_wheel = None


# See https://github.com/bigartm/bigartm/issues/840
class install_plat_lib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


setup(
    name="xmos_ai_tools",
    use_scm_version={
        "root": "../",
        "relative_to": __file__,
        "local_scheme": "no-local-version",
    },
    setup_requires=["setuptools_scm"],
    author="XMOS",
    author_email="support@xmos.com",
    license="LICENSE.txt",
    description="XMOS AI Tools",
    long_description=LONG_README,
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
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    package_data={"xmos_ai_tools.xinterpreters.host": XTFLM_INTERPRETER_LIBS},
    data_files=[
        ("Scripts" if platform.system() == "Windows" else "bin", [XCOREOPT_BINARY])
    ],
    cmdclass={
        "bdist_wheel": bdist_wheel,
        "install": install_plat_lib,
    },
    keywords="tensorflow binarized neural networks",
)
