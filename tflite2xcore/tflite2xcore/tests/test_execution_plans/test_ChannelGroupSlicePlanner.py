# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging

import numpy as np

from tflite2xcore.execution_planning import ChannelGroupSlicePlanner
from tflite2xcore.xlogging import LoggingContext


MAX_OUTPUT_CHANNELS = 75
VALID_OUTPUT_CHANNELS = list(range(1, MAX_OUTPUT_CHANNELS + 1))
VALID_NUM_THREAD = list(range(1, ChannelGroupSlicePlanner.MAX_THREADS + 1))


@pytest.mark.parametrize("num_channels", VALID_OUTPUT_CHANNELS)
def test_channel_coverage(num_channels):
    planner = ChannelGroupSlicePlanner(
        num_channels, num_threads=ChannelGroupSlicePlanner.MAX_THREADS
    )
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        coverage_map = np.zeros(num_channels, dtype=bool)
        for changrp in plan.changrp_slices:
            Cbegin, Cend = changrp
            coverage_map[Cbegin : Cend + 1] = True
        assert np.all(coverage_map)


def generate_thread_cost_array(max_channels=MAX_OUTPUT_CHANNELS):
    thread_costs = np.zeros(
        (max_channels, ChannelGroupSlicePlanner.MAX_THREADS), dtype=np.float
    )

    for num_channels in range(1, MAX_OUTPUT_CHANNELS + 1):
        for num_threads in VALID_NUM_THREAD:
            planner = ChannelGroupSlicePlanner(
                num_channels, num_threads=num_threads, forced=True
            )
            plan = planner.find_optimal_plan()
            print(num_channels, num_threads, plan)
            thread_costs[num_channels - 1, num_threads - 1] = plan.cost

    return thread_costs


@pytest.fixture(scope="session")
def thread_cost_array():
    return generate_thread_cost_array()


@pytest.mark.parametrize("num_channels", VALID_OUTPUT_CHANNELS)
def test_optimal_thread_count(num_channels, thread_cost_array):
    planner = ChannelGroupSlicePlanner(
        num_channels, num_threads=ChannelGroupSlicePlanner.MAX_THREADS
    )
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[num_channels - 1, :]
    assert np.min(costs) == plan.cost
    assert np.argmin(costs) == plan.num_threads - 1


if __name__ == "__main__":
    pytest.main()
