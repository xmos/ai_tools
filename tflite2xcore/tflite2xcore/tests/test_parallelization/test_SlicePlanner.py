# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging
import itertools

import numpy as np

from tflite2xcore.parallelization import SlicePlanner, MAX_THREADS
from tflite2xcore.xlogging import LoggingContext

MAX_OUTPUT_CHANNELS = 20
VALID_OUTPUT_CHANNELS = list(range(1, MAX_OUTPUT_CHANNELS + 1))
VALID_NUM_THREAD = list(range(1, MAX_THREADS + 1))

MAX_HEIGHT = MAX_WIDTH = 10
VALID_HEIGHT = list(range(1, MAX_HEIGHT + 1))
VALID_WIDTH = list(range(1, MAX_WIDTH + 1))
VALID_NUM_THREAD = list(range(1, MAX_THREADS + 1))


@pytest.mark.parametrize("num_channels", VALID_OUTPUT_CHANNELS)
@pytest.mark.parametrize("height", VALID_HEIGHT)
@pytest.mark.parametrize("width", VALID_WIDTH)
def test_layout_coverage(num_channels, height, width):
    planner = SlicePlanner(num_channels, height, width, num_threads=MAX_THREADS)
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros((height, width), dtype=bool)
        for block in plan.rowcol_slices:
            y_start, y_end = block[0], block[0] + block[2]
            x_start, x_end = block[1], block[1] + block[3]
            coverage_map[y_start:y_end, x_start:x_end] = True
        assert np.all(coverage_map)

        coverage_map = np.zeros(num_channels, dtype=bool)
        for block in plan.changrp_slices:
            Cbegin, Cend = block
            coverage_map[Cbegin : Cend + 1] = True
        assert np.all(coverage_map)


def generate_thread_cost_array(
    max_channel=MAX_OUTPUT_CHANNELS, max_height=MAX_HEIGHT, max_width=MAX_WIDTH
):
    thread_costs = np.zeros(
        (max_channel, max_height, max_width, MAX_THREADS), dtype=np.float
    )

    for c, y, x in itertools.product(
        range(max_channel), range(max_height), range(max_width)
    ):
        for num_threads in VALID_NUM_THREAD:
            planner = SlicePlanner(
                Cout=c + 1,
                height=y + 1,
                width=x + 1,
                num_threads=num_threads,
                forced=True,
            )
            plan = planner.find_optimal_plan()
            thread_costs[c, y, x, num_threads - 1] = plan.cost

    return thread_costs


@pytest.fixture(scope="session")
def thread_cost_array():
    return generate_thread_cost_array()


@pytest.mark.parametrize("num_channels", VALID_OUTPUT_CHANNELS)
@pytest.mark.parametrize("height", VALID_HEIGHT)
@pytest.mark.parametrize("width", VALID_WIDTH)
def test_optimal_thread_count(num_channels, height, width, thread_cost_array):
    planner = SlicePlanner(num_channels, height, width, num_threads=MAX_THREADS)
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[num_channels - 1, height - 1, width - 1, :]
    assert np.min(costs) == plan.cost
    assert np.argmin(costs) == plan.num_threads - 1


if __name__ == "__main__":
    pytest.main()
