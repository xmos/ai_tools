# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--test-file',
        action='store',
        default='all_tests.txt',
        help="Path to test file"
    )
    parser.addoption(
        '--test-app',
        action='store',
        default='../examples/apps/test_model/bin/test_model.xe',
        help="Path to test_model or test_model.xe"
    )


@pytest.fixture
def test_model_app(request):
    test_model_app = request.config.getoption('--test-app')
    if not os.path.exists(test_model_app):
        print(f'ABORTING: {test_model_app} does not exist!')
        exit()
    return test_model_app

