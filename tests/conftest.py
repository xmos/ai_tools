# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import pytest

import directories

def pytest_addoption(parser):
    parser.addoption(
        '--test-dir',
        action='store',
        default=directories.OP_TEST_MODELS_DATA_DIR,
        help="Path to test data"
    )

    parser.addoption(
        '--test-app',
        action='store',
        default=None,
        help="Path to test_model or test_model.xe"
    )

    parser.addoption(
        '--max-count',
        action='store',
        type='int',
        default=sys.maxsize,
        help="Maximum number of tests per operator to run"
    )

    parser.addoption(
        '--abs-tol',
        type='int',
        default=1,
        help="Maximum allowable absolute difference between computed and expected results."
    )


@pytest.fixture
def test_model_app(request):
    test_model_app = request.config.getoption('--test-app')
    if test_model_app and not os.path.exists(test_model_app):
        print(f'ABORTING: {test_model_app} does not exist!')
        exit()
    return test_model_app


@pytest.fixture
def abs_tol(request):
    abs_tol = request.config.getoption('--abs-tol')
    return abs_tol
