#!/usr/bin/python3

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import platform
from setuptools import setup
from setuptools.command.install import install
import pathlib
import os

# Find path to xcore-opt binary
here = pathlib.Path(__file__).parent.resolve()
exe_suffix = ".exe" if platform.system() == "Windows" else ""
XCOREOPT_BINARY = pathlib.Path.joinpath(here.parent, "bazel-bin", "xcore-opt",
                                        exe_suffix)

# Get the long description from the README file
LONG_README = (here / 'README.md').read_text(encoding='utf-8')

# tflm_interpreter path and libs from lib_tflite_micro
TFLM_INTERPRETER_LIBS = [
    "/libs/linux/tflm_python.so",
    "/libs/linux/tflm_python.so.1.0.1",
    "/libs/macos/tflm_python.dylib",
    "/libs/macos/tflm_python.1.0.1.dylib",
]
TFLM_INTERPRETER_PATH = pathlib.Path.joinpath(here.parent.parent.parent, "third_party", "lib_tflite_micro", "tflm_interpreter", "tflm_interpreter")
# adjust path to libs
TFLM_INTERPRETER_LIBS = [str(TFLM_INTERPRETER_PATH) + x for x in TFLM_INTERPRETER_LIBS]
# tflm_interpreter requires numpy
REQUIRED_PACKAGES = [
    "numpy<2.0",
]

# Get tag version from env variable
# This will be in the format, vX.Y.Z
# We need to remove the first character to get just the version number
environment_variable_name = 'XMOS_AI_TOOLS_RELEASE_VERSION'
VERSION_NUMBER = os.environ.get( environment_variable_name, "v0.1.0" )
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


# See https://github.com/bigartm/bigartm/issues/840
class install_plat_lib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


setup(
    name="xmos-ai-tools",
    version=VERSION_NUMBER,
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
    install_requires=REQUIRED_PACKAGES,
    package_dir={'xmos_ai_tools.xformer': 'src/xformer', 'xmos_ai_tools.tflm_interpreter': str(TFLM_INTERPRETER_PATH)},
    packages=['xmos_ai_tools.xformer', 'xmos_ai_tools.tflm_interpreter'],  # Required
    package_data={"": TFLM_INTERPRETER_LIBS},
    data_files=[('bin', [str(XCOREOPT_BINARY)])],
    cmdclass={
        'bdist_wheel': bdist_wheel,
        'install': install_plat_lib,
    },
    keywords="tensorflow binarized neural networks",
)
