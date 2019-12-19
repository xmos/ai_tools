import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--test-model',
        action='store',
        default='../examples/test_model/bin/test_model.xe',
        help="Path to test_model or test_model.xe"
    )


@pytest.fixture
def test_model_app(request):
    test_model_app = request.config.getoption('--test-model')
    if not os.path.exists(test_model_app):
        print(f'ABORTING: {test_model_app} does not exist!')
        exit()
    return test_model_app

