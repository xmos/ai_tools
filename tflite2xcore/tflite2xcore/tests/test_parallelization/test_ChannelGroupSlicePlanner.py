# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest

import numpy as np

from tflite2xcore.parallelization import ChannelGroupSlicePlanner, MAX_THREADS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MAX_OUTPUT_CHANNELS = 75
VALID_OUTPUT_CHANNELS = list(range(1, MAX_OUTPUT_CHANNELS + 1))
VALID_NUM_THREAD = list(range(1, MAX_THREADS + 1))

PARAMS = {"default": {"num_channels": list(range(1, MAX_OUTPUT_CHANNELS + 1))}}

@pytest.mark.parametrize("num_channels", VALID_OUTPUT_CHANNELS)
def test_channel_coverage(num_channels):
    planner = ChannelGroupSlicePlanner(num_channels, num_threads=MAX_THREADS)
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros(num_channels, dtype=bool)
        for changrp in plan.changrp_slices:
            Cbegin, Cend = changrp
            coverage_map[Cbegin : Cend + 1] = True
        assert np.all(coverage_map)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


def generate_thread_cost_array(max_channels: int = MAX_OUTPUT_CHANNELS) -> np.ndarray:
    thread_costs = np.zeros((max_channels, MAX_THREADS), dtype=np.float)

    for num_channels in range(1, MAX_OUTPUT_CHANNELS + 1):
        for num_threads in list(range(1, MAX_THREADS + 1)):
            planner = ChannelGroupSlicePlanner(
                num_channels, num_threads=num_threads, forced=True
            )
            plan = planner.find_optimal_plan()
            thread_costs[num_channels - 1, num_threads - 1] = plan.estimate_cost()

    return thread_costs


@pytest.fixture(scope="session")  # type: ignore
def thread_cost_array() -> np.ndarray:
    return generate_thread_cost_array()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_channel_coverage(num_channels: int) -> None:
    planner = ChannelGroupSlicePlanner(num_channels, num_threads=MAX_THREADS)
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros(num_channels, dtype=bool)
        for changrp in plan._channel_groups:
            coverage_map[changrp.begin : changrp.end + 1] = True
        assert np.all(coverage_map)


def test_optimal_thread_count(num_channels: int, thread_cost_array: np.ndarray) -> None:
    planner = ChannelGroupSlicePlanner(num_channels, num_threads=MAX_THREADS)
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[num_channels - 1, :]
    assert np.min(costs) == plan.estimate_cost()
    assert np.argmin(costs) == plan._num_threads - 1


if __name__ == "__main__":
    pytest.main()
