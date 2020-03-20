# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import numbers

from abc import ABC, abstractmethod


class ParPlan():
    def __init__(self, num_threads, layout, cost):
        self.num_threads = num_threads

        if isinstance(cost, numbers.Number):
            self.cost = cost
        else:
            assert hasattr(cost, '__call__')
            self.cost = cost(layout)
        self.layout = layout  # should be a list of 4-tuples

    def __repr__(self):
        return f"{type(self).__name__} (num_threads={self.num_threads}, cost={self.cost})"

    def details(self):
        return f"{repr(self)}, with layout {self.layout}"


class ParallelizationPlanner(ABC):
    MAX_THREADS = 5

    def __init__(self, *, num_threads, forced=False):
        assert isinstance(num_threads, int)
        assert 0 < num_threads <= self.MAX_THREADS
        self.logger = logging.getLogger(self.__class__.__name__)
        if num_threads == 1:
            self.logger.warning(f"initialized with 1 thread.")
        self.num_threads = num_threads

        self.forced = forced
        self._candidate_plans = []  # should be a list of ParPlans

    @abstractmethod
    def create_n_thread_candidates(self, num_threads):
        pass

    @abstractmethod
    def estimate_block_cost(self, y_start, x_start, y_width, x_width):
        pass

    @abstractmethod
    def estimate_layout_cost(self, layout):
        pass

    def add_layout_candidate(self, *args):
        self._candidate_plans.append(
            ParPlan(*args, cost=lambda layout: self.estimate_layout_cost(layout))  # pylint: disable=no-value-for-parameter
        )

    def create_candidate_plans(self):
        for n in range(self.num_threads):
            self.create_n_thread_candidates(n + 1)

    def find_optimal_plan(self):
        if not self._candidate_plans:
            self.create_candidate_plans()

        best_plan = min(self._candidate_plans, key=lambda plan: plan.cost)
        if best_plan.num_threads == self.num_threads:
            self.logger.debug(f"found best plan: {repr(best_plan)}")
            return best_plan
        else:
            forced_candidates = [plan for plan in self._candidate_plans
                                 if plan.num_threads == self.num_threads]
            best_forced_plan = min(forced_candidates, key=lambda plan: plan.cost)

        if self.forced:
            self.logger.warning(
                f"forcing suboptimal plan {repr(best_forced_plan)} "
                f"when better alternative {repr(best_plan)} exists."
            )
            return best_forced_plan
        else:
            self.logger.info(
                f"replacing suboptimal plan {repr(best_forced_plan)} "
                f"with better alternative {repr(best_plan)}."
            )
            return best_plan


class UnidirectionalSplitPlanner(ParallelizationPlanner):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(height, int), f"received height={height} with type {type(height)}"
        assert isinstance(width, int), f"received width={width} with type {type(width)}"
        assert height * width > 0, f"received height={height}, width={width}"
        self.height, self.width = height, width

    _adjustments = {
        1: lambda rem: [0],
        2: lambda rem: [int(rem >= 1), 0],
        3: lambda rem: [int(rem >= 1), 0, int(rem >= 2)],
        4: lambda rem: [int(rem >= 1), int(rem == 3), 0, int(rem >= 2)],
        5: lambda rem: [int(rem >= 1), int(rem >= 3), 0, int(rem >= 4), int(rem >= 2)]
    }

    def unidir_split_helper(self, dim, num_threads):
        base, rem = dim // num_threads, dim % num_threads
        block_lengths = [base + a for a in self._adjustments[num_threads](rem)]
        block_starts = [0]
        for j in range(num_threads - 1):
            block_starts.append(block_starts[j] + block_lengths[j])
        return block_starts, block_lengths

    def unidir_width_layout(self, num_threads):
        starts, widths = self.unidir_split_helper(self.width, num_threads)
        return [(0, starts[j], self.height, widths[j]) for j in range(num_threads)]

    def unidir_height_layout(self, num_threads):
        starts, heights = self.unidir_split_helper(self.height, num_threads)
        return [(starts[j], 0, heights[j], self.width) for j in range(num_threads)]

    def create_n_thread_candidates(self, num_threads):
        self.add_layout_candidate(
            num_threads, self.unidir_width_layout(num_threads))
        if num_threads > 1 and self.width != self.height:
            self.add_layout_candidate(
                num_threads, self.unidir_height_layout(num_threads))


class DIDOConv2DPlanner(UnidirectionalSplitPlanner):
    def __init__(self, height, width, *, padding='VALID', **kwargs):
        super().__init__(height, width, **kwargs)
        assert padding in ['VALID', 'SAME']
        self.padding = padding  # TODO: currently unused

    def estimate_block_cost(self, y_start, x_start, y_width, x_width):
        return y_width * x_width

    def estimate_layout_cost(self, layout):
        return max(self.estimate_block_cost(*block) for block in layout)

    def create_n_thread_candidates(self, num_threads):
        super().create_n_thread_candidates(num_threads)
        if num_threads == 3:
            # TODO: implement me
            # idea: one wide block with three edges and two corner blocks
            # do this with both horizontal and vertical orientation
            pass
        if num_threads == 4:
            # split into approximate quarters
            bh, bw = self.height // 2, self.width // 2  # block heights and widths
            rh, rw = self.height % 2, self.width % 2
            layout = [(0,  0,  bh,      bw),
                      (0,  bw, bh,      bw + rw),
                      (bh, 0,  bh + rh, bw),
                      (bh, bw, bh + rh, bw + rw)]
            self.add_layout_candidate(num_threads, layout)
        if num_threads == 5:
            # TODO: implement me
            # idea: two blocks on opposite ends, each with two corners and three edges
            #       then two blocks between these, each with one edge
            #       and one block without any edges
            # do this with both horizontal and vertical orientation
            pass
