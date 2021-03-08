# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest
import itertools

import numpy as np

from tflite2xcore.parallelization import SlicePlanner, MAX_THREADS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MAX_OUTPUT_CHANNELS = 20
MAX_HEIGHT = MAX_WIDTH = 10

PARAMS = {
    "default": {
        "num_channels": list(range(1, MAX_OUTPUT_CHANNELS + 1)),
        "height": list(range(1, MAX_HEIGHT + 1)),
        "width": list(range(1, MAX_WIDTH + 1)),
    }
}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


def generate_thread_cost_array(
    max_channel: int = MAX_OUTPUT_CHANNELS,
    max_height: int = MAX_HEIGHT,
    max_width: int = MAX_WIDTH,
) -> np.ndarray:
    thread_costs = np.zeros(
        (max_channel, max_height, max_width, MAX_THREADS), dtype=np.float
    )

    for c, y, x in itertools.product(
        range(max_channel), range(max_height), range(max_width)
    ):
        for num_threads in list(range(1, MAX_THREADS + 1)):
            planner = SlicePlanner(
                num_channels_out=c + 1,
                height=y + 1,
                width=x + 1,
                num_threads=num_threads,
                forced=True,
            )
            plan = planner.find_optimal_plan()
            thread_costs[c, y, x, num_threads - 1] = plan.estimate_cost()

    return thread_costs


@pytest.fixture(scope="session")  # type: ignore
def thread_cost_array() -> np.ndarray:
    return generate_thread_cost_array()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_layout_coverage(num_channels: int, height: int, width: int) -> None:
    planner = SlicePlanner(num_channels, height, width, num_threads=MAX_THREADS)
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros((height, width), dtype=bool)
        for block in plan._row_col_slices:
            y_start, y_end = block.top, block.top + block.rows
            x_start, x_end = block.left, block.left + block.cols
            coverage_map[y_start:y_end, x_start:x_end] = True
        assert np.all(coverage_map)

        coverage_map = np.zeros(num_channels, dtype=bool)
        for changrp in plan._channel_groups:
            coverage_map[changrp.begin : changrp.end + 1] = True
        assert np.all(coverage_map)


def test_optimal_thread_count(
    num_channels: int, height: int, width: int, thread_cost_array: np.ndarray
) -> None:
    planner = SlicePlanner(num_channels, height, width, num_threads=MAX_THREADS)
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[num_channels - 1, height - 1, width - 1, :]
    assert np.min(costs) == plan.estimate_cost()
    assert np.argmin(costs) == plan._num_threads - 1


if __name__ == "__main__":
    pytest.main()
