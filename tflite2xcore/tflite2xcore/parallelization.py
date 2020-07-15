# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numbers
import math
import enum

from abc import ABC, abstractmethod

from tflite2xcore import xlogging as logging

MAX_THREADS = 5
CHANNEL_GROUP_SIZE = 16


class ParallelizationPlan:
    def __init__(
        self, num_threads, *, cost, changrp_slices=None, rowcol_slices=None,
    ):
        self.num_threads = num_threads
        self.changrp_slices = changrp_slices  # should be a list of 2-tuples (start channel index, end channel index)
        self.rowcol_slices = (
            rowcol_slices  # should be a list of 4-tuples (top, left, rows, cols)
        )

        if isinstance(cost, numbers.Number):
            self.cost = cost
        else:
            assert hasattr(cost, "__call__")
            self.cost = cost(self)

    def __repr__(self):
        return (
            f"{type(self).__name__} (num_threads={self.num_threads}, cost={self.cost})"
        )

    def to_dict(self):
        bits = {"th": self.num_threads}
        if self.changrp_slices is not None:
            bits["cg"] = self.changrp_slices
        if self.rowcol_slices is not None:
            bits["rc"] = self.rowcol_slices

        return bits


class ParallelizationPlanner(ABC):
    def __init__(self, *, num_threads, forced=False):
        assert isinstance(num_threads, int)
        assert 0 < num_threads <= MAX_THREADS
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_threads = num_threads

        self.forced = forced
        self._candidate_plans = []  # should be a list of ParallizationPlans

    @abstractmethod
    def create_n_thread_candidates(self, num_threads):
        pass

    @abstractmethod
    def estimate_plan_cost(self, plan):
        pass

    def add_candidate_plan(self, plan):
        self._candidate_plans.append(plan)

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
            forced_candidates = [
                plan
                for plan in self._candidate_plans
                if plan.num_threads == self.num_threads
            ]
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
        assert isinstance(
            height, int
        ), f"received height={height} with type {type(height)}"
        assert isinstance(width, int), f"received width={width} with type {type(width)}"
        assert height * width > 0, f"received height={height}, width={width}"
        self.height, self.width = height, width

    @staticmethod
    def unidir_split_helper(dim, num_threads):
        adjustments = {
            1: lambda rem: [0],
            2: lambda rem: [int(rem >= 1), 0],
            3: lambda rem: [int(rem >= 1), 0, int(rem >= 2)],
            4: lambda rem: [int(rem >= 1), int(rem == 3), 0, int(rem >= 2)],
            5: lambda rem: [
                int(rem >= 1),
                int(rem >= 3),
                0,
                int(rem >= 4),
                int(rem >= 2),
            ],
        }

        base, rem = dim // num_threads, dim % num_threads
        block_lengths = [base + a for a in adjustments[num_threads](rem)]
        block_starts = [0]
        for j in range(num_threads - 1):
            block_starts.append(block_starts[j] + block_lengths[j])
        return block_starts, block_lengths

    def unidir_width_layout(self, num_threads):
        starts, widths = UnidirectionalSplitPlanner.unidir_split_helper(
            self.width, num_threads
        )
        return [[0, starts[j], self.height, widths[j]] for j in range(num_threads)]

    def unidir_height_layout(self, num_threads):
        starts, heights = UnidirectionalSplitPlanner.unidir_split_helper(
            self.height, num_threads
        )
        return [[starts[j], 0, heights[j], self.width] for j in range(num_threads)]


class ChannelGroupSlicePlanner(ParallelizationPlanner):
    def __init__(self, Cout, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(Cout, int), f"received Cout={Cout} with type {type(Cout)}"
        assert Cout > 0, f"received Cout={Cout}"
        self.Cout = Cout

    @staticmethod
    def changrp_split_helper(num_channels):
        changrps = []
        num_changrps = math.ceil(num_channels / float(CHANNEL_GROUP_SIZE))
        for i in range(num_changrps):
            Cbegin = i * CHANNEL_GROUP_SIZE
            Cend = min(Cbegin + CHANNEL_GROUP_SIZE - 1, num_channels - 1,)
            changrps.append([Cbegin, Cend])

        return changrps

    def estimate_plan_cost(self, plan):
        def estimate_changrp_cost(changrp):
            Cbegin, Cend = changrp
            if Cend - Cbegin + 1 == CHANNEL_GROUP_SIZE:
                return 1
            else:
                return 2  # NOTE: 2 might be a bit aggressive

        return (
            sum(estimate_changrp_cost(changrp) for changrp in plan.changrp_slices)
            / plan.num_threads
        )

    def create_n_thread_candidates(self, num_threads):
        changrps = ChannelGroupSlicePlanner.changrp_split_helper(self.Cout)
        if len(changrps) < num_threads:
            self.logger.info(
                f"create_n_thread_candidates: num_threads={num_threads}, changrps={str(changrps)}"
            )
            self.add_candidate_plan(
                ParallelizationPlan(
                    num_threads,
                    cost=lambda plan: self.estimate_plan_cost(plan),
                    changrp_slices=changrps,
                )
            )


class SlicePlanner(ParallelizationPlanner):
    def __init__(self, Cout, height, width, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(Cout, int), f"received Cout={Cout} with type {type(Cout)}"
        assert Cout > 0, f"received Cout={Cout}"
        assert isinstance(
            height, int
        ), f"received height={height} with type {type(height)}"
        assert isinstance(width, int), f"received width={width} with type {type(width)}"
        assert height * width > 0, f"received height={height}, width={width}"
        self.Cout = Cout
        self.height, self.width = height, width

    def estimate_plan_cost(self, plan):
        def estimate_changrp_cost(changrp):
            Cbegin, Cend = changrp
            if Cend - Cbegin + 1 == CHANNEL_GROUP_SIZE:
                return 1
            else:
                return 2  # NOTE: 2 might be a bit aggressive

        def estimate_row_slice_cost(row_slice):
            _, _, y_width, x_width = row_slice  # first two items are y_start, x_start
            return y_width * x_width

        cost = 0
        for changrp_slice in plan.changrp_slices:
            changrp_cost = estimate_changrp_cost(changrp_slice)
            for row_slice in plan.rowcol_slices:
                row_slice_cost = estimate_row_slice_cost(row_slice)
            cost += changrp_cost * row_slice_cost

        return cost

    def create_n_thread_candidates(self, num_threads):

        starts, heights = UnidirectionalSplitPlanner.unidir_split_helper(
            self.height, num_threads
        )

        row_slices = [
            [starts[j], 0, heights[j], self.width] for j in range(num_threads)
        ]
        changrps = ChannelGroupSlicePlanner.changrp_split_helper(self.Cout)

        self.logger.info(
            f"create_n_thread_candidates: num_threads={num_threads}, changrps={str(changrps)}, row_slices={str(row_slices)}"
        )
        self.add_candidate_plan(
            ParallelizationPlan(
                num_threads,
                cost=lambda plan: self.estimate_plan_cost(plan),
                changrp_slices=changrps,
                rowcol_slices=row_slices,
            )
        )
