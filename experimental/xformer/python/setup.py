#!/usr/bin/python3

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import os
import platform
import setuptools

XFORMER_BINARY = [
    "tflm_python.dylib",
]

EXCLUDES = ["xformer.py"]

README = r'''
XMOS AI Tools
'''

exe_suffix = ".exe" if platform.system() == "Windows" else ""

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


setuptools.setup(
    name="xformer",
    version="1.0",
    author="XMOS",
    author_email="support@xmos.com",
    license="LICENSE.txt",
    description="IREE TFLite Compiler Tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/google/iree",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.7",
    packages=["xmosaitools"],
    package_data={"xformer": XFORMER_BINARY},
    cmdclass={
        'bdist_wheel': bdist_wheel,
    },
    include_package_data=True,
    zip_safe=False,  # This package is fine but not zipping is more versatile.
    use_scm_version={
        "root": "..",
        "relative_to": __file__,
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
)
