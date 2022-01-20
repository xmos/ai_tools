#!/usr/bin/python3

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import platform
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

exe_suffix = ".exe" if platform.system() == "Windows" else ""
XCOREOPT_BINARY = pathlib.Path.joinpath(here.parent, "bazel-bin", "xcore-opt",
                                        exe_suffix)

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
    name="xmosaitools",
    version="1.0",
    author="XMOS",
    author_email="support@xmos.com",
    license="LICENSE.txt",
    description="XMOS AI Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xmos/ai_tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.7",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    data_files=[('bin', [str(XCOREOPT_BINARY)])],
    cmdclass={
        'bdist_wheel': bdist_wheel,
    },
)
