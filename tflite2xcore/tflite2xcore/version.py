# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1
from contextlib import suppress


def get_version() -> str:
    with suppress(Exception):
        try:
            # use setuptools_scm if installed
            #   setuptools_scm will append commit info the base version number
            from setuptools_scm import get_version

            return get_version(
                root="../..", relative_to=__file__, version_scheme="post-release"
            )
        except ImportError:
            # fall back to the builtin importlib_metadata module
            #   importlib_metadata returns the version number in the package metadata
            from importlib_metadata import version

            try:
                return version(__name__)
            except:
                return "Unable to determine version from package."
