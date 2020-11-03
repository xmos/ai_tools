# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved


def get_version() -> str:
    try:
        # use setuptools_scm if installed
        #   setuptools_scm will append commit info the base version number
        from setuptools_scm import get_version

        return get_version(
            root="../..", relative_to=__file__, version_scheme="post-release"
        )
    except:
        # fall back to the builtin importlib_metadata module
        #   importlib_metadata returns the version number in the package metadata
        from importlib_metadata import version

        return version(__name__)
