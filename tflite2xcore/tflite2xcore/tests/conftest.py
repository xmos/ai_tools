# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


#  ----------------------------------------------------------------------------
#                                CONFIG FLAGS
#  ----------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", help="smoke test")


#  ----------------------------------------------------------------------------
#                                   HOOKS
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    try:
        PARAMS = metafunc.module.PARAMS
        if metafunc.config.getoption("smoke"):
            params = PARAMS["smoke"]
        else:
            params = PARAMS["default"]
    except AttributeError:
        params = {}

    for name, values in params.items():
        if name in metafunc.fixturenames:
            metafunc.parametrize(name, values)
