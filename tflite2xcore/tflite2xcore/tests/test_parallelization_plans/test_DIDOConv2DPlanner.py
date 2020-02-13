# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging
import pathlib
import argparse
import itertools
import numpy as np

from tflite2xcore.parallelization import DIDOConv2DPlanner
from tflite2xcore.utils import LoggingContext


__DIR_PATH = pathlib.Path(__file__).parent
REGRESSION_DATA_PATH = __DIR_PATH.joinpath('regression_data', 'DIDOConv2DPlanner.npz').resolve()

MAX_HEIGHT = MAX_WIDTH = 32
VALID_HEIGHT = list(range(1, MAX_HEIGHT + 1))
VALID_WIDTH = list(range(1, MAX_WIDTH + 1))
VALID_NUM_THREAD = list(range(1, DIDOConv2DPlanner.MAX_THREADS + 1))


@pytest.mark.parametrize('height', VALID_HEIGHT)
@pytest.mark.parametrize('width', VALID_WIDTH)
def test_layout_coverage(height, width):
    planner = DIDOConv2DPlanner(height, width,
                                num_threads=DIDOConv2DPlanner.MAX_THREADS)
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros((height, width), dtype=bool)
        for block in plan.layout:
            y_start, y_end = block[0], block[0] + block[2]
            x_start, x_end = block[1], block[1] + block[3]
            coverage_map[y_start:y_end, x_start:x_end] = True
        assert np.all(coverage_map)


def generate_thread_cost_array(max_height=MAX_HEIGHT, max_width=MAX_WIDTH):
    thread_costs = np.zeros(
        (max_height, max_width, DIDOConv2DPlanner.MAX_THREADS), dtype=np.int32)

    for y, x in itertools.product(range(max_height), range(max_width)):
        for num_threads in VALID_NUM_THREAD:
            planner = DIDOConv2DPlanner(height=y + 1, width=x + 1,
                                        num_threads=num_threads, forced=True)
            plan = planner.find_optimal_plan()
            thread_costs[y, x, num_threads - 1] = plan.cost

    return thread_costs


@pytest.fixture(scope='session')
def thread_cost_array():
    return generate_thread_cost_array()


@pytest.mark.parametrize('height', VALID_HEIGHT)
@pytest.mark.parametrize('width', VALID_WIDTH)
def test_optimal_thread_count(height, width, thread_cost_array):
    planner = DIDOConv2DPlanner(height, width,
                                num_threads=DIDOConv2DPlanner.MAX_THREADS)
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[height - 1, width - 1, :]
    assert np.min(costs) == plan.cost
    assert np.argmin(costs) == plan.num_threads - 1


@pytest.fixture(scope='session')
def regression_data():
    return np.load(REGRESSION_DATA_PATH)


@pytest.mark.parametrize('num_threads', VALID_NUM_THREAD)
def test_regression(regression_data, thread_cost_array, num_threads):
    current = thread_cost_array[:, :, num_threads - 1]
    reference = regression_data['thread_cost_array'][:, :, num_threads - 1]
    assert np.all(current == reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--update_regression_data', action='store_true', default=False,
        help='Recalculate and overwrite the regression data file.')
    args = parser.parse_args()

    if args.update_regression_data:
        logging.warning(f"Updating regression data at {REGRESSION_DATA_PATH}")
        with LoggingContext(logging.getLogger(), logging.ERROR):
            data = {'thread_cost_array': generate_thread_cost_array()}
        np.savez(REGRESSION_DATA_PATH, **data)
    else:
        pytest.main()
