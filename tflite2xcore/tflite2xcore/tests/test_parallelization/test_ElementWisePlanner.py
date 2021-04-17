# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

import numpy as np

from tflite2xcore.parallelization import ElementWisePlanner, MAX_THREADS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MAX_ELEMENTS = 75

PARAMS = {"default": {"num_elements": list(range(1, MAX_ELEMENTS + 1))}}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


def generate_thread_cost_array(max_elements: int = MAX_ELEMENTS) -> np.ndarray:
    thread_costs = np.zeros((max_elements, MAX_THREADS), dtype=np.float)

    for num_elements in range(1, max_elements + 1):
        for num_threads in list(range(1, MAX_THREADS + 1)):
            planner = ElementWisePlanner(
                num_elements,
                num_threads=num_threads,
                forced=True,
                fixed_cost_per_thread=10,
            )
            plan = planner.find_optimal_plan()
            thread_costs[num_elements - 1, num_threads - 1] = plan.estimate_cost()

    return thread_costs


@pytest.fixture(scope="session")  # type: ignore
def thread_cost_array() -> np.ndarray:
    return generate_thread_cost_array()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_element_coverage(num_elements: int) -> None:
    planner = ElementWisePlanner(
        num_elements, num_threads=MAX_THREADS, fixed_cost_per_thread=10
    )
    planner.create_candidate_plans()
    for plan in planner._candidate_plans:
        assert num_elements == sum(plan._job_sizes)


def test_optimal_thread_count(num_elements: int, thread_cost_array: np.ndarray) -> None:
    planner = ElementWisePlanner(
        num_elements, num_threads=MAX_THREADS, fixed_cost_per_thread=10
    )
    plan = planner.find_optimal_plan()
    costs = thread_cost_array[num_elements - 1, :]
    assert np.min(costs) == plan.estimate_cost()
    assert np.argmin(costs) == plan._num_threads - 1


if __name__ == "__main__":
    pytest.main()
