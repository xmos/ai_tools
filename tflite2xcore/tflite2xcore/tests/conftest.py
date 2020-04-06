# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging


#  ----------------------------------------------------------------------------
#                                   HOOKS
#  ----------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", help="smoke test")
    parser.addoption("--extended", action="store_true", help="extended test")    


# TODO: this is deprecated, find a better way
def pytest_cmdline_preparse(config, args):
    if "--smoke" in args and "--extended" in args:
        raise pytest.UsageError('Only one of "--smoke" and "--extended" can be used')


def pytest_generate_tests(metafunc):
    try:
        PARAMS = metafunc.module.PARAMS
        if metafunc.config.getoption("smoke"):
            params = PARAMS.get("smoke", PARAMS["default"])
        elif metafunc.config.getoption("extended"):
            params = PARAMS.get("extended", PARAMS["default"])
        else:
            params = PARAMS["default"]
    except AttributeError:
        params = {}

    for name, values in params.items():
        if name in metafunc.fixturenames:
            metafunc.parametrize(name, values)
