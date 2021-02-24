# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import math
import logging
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    Callable,
    List,
    Tuple,
    Optional,
    SupportsFloat,
    NamedTuple,
    Generic,
    TypeVar,
    Sequence,
)

from tflite2xcore.utils import ACC_PERIOD_INT8, VECTOR_SIZE_BYTES

MAX_THREADS = 5
CHANNEL_GROUP_SIZE = ACC_PERIOD_INT8


class ParallelizationPlan(ABC):
    def __init__(
        self, num_threads: int, *, fixed_cost_per_thread: SupportsFloat
    ) -> None:
        self._num_threads = num_threads
        self._fixed_cost_per_thread = fixed_cost_per_thread

    def estimate_fixed_cost(self) -> float:
        return self._num_threads * float(self._fixed_cost_per_thread)

    @abstractmethod
    def estimate_cost(self) -> SupportsFloat:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__} (num_threads={self._num_threads}, cost={self.estimate_cost()})"

    def to_dict(self) -> Dict[str, Any]:
        return {"th": self._num_threads}


class ElementWiseParallelizationPlan(ParallelizationPlan):
    def __init__(
        self,
        num_threads: int,
        *,
        job_sizes: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_threads, **kwargs)
        self._job_sizes = list(job_sizes or [])

    def estimate_cost(self) -> SupportsFloat:
        return max(self._job_sizes) + self.estimate_fixed_cost()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self._job_sizes:
            d["eg"] = self._job_sizes
        return d


class _ChannelGroup(NamedTuple):
    begin: int
    end: int


class ChannelGroupParallelizationPlan(ParallelizationPlan):
    def __init__(
        self,
        num_threads: int,
        *,
        channel_groups: Optional[Sequence[_ChannelGroup]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_threads, **kwargs)
        self._channel_groups = list(channel_groups or [])

    def _estimate_channel_group_cost(self, changrp: _ChannelGroup) -> int:
        if changrp.begin - changrp.begin + 1 == CHANNEL_GROUP_SIZE:
            return 1
        else:
            return 2  # NOTE: 2 might be a bit aggressive

    def estimate_cost(self) -> SupportsFloat:
        return (
            sum(
                self._estimate_channel_group_cost(changrp)
                for changrp in self._channel_groups
            )
            / self._num_threads
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self._channel_groups:
            d["cg"] = [tuple(t) for t in self._channel_groups]
        return d


class _RowColumnSlice(NamedTuple):
    top: int
    left: int
    rows: int
    cols: int


class RowColumnParallelizationPlan(ChannelGroupParallelizationPlan):
    def __init__(
        self,
        num_threads: int,
        *,
        row_column_slices: Optional[Sequence[_RowColumnSlice]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_threads, **kwargs)
        self._row_col_slices = row_column_slices or []

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self._row_col_slices is not None:
            d["rc"] = [tuple(t) for t in self._row_col_slices]
        return d

    def _estimate_row_slice_cost(self, row_col_slice: _RowColumnSlice) -> int:
        return row_col_slice.rows * row_col_slice.cols

    def estimate_cost(self) -> SupportsFloat:
        cost = 0
        for changrp_slice in self._channel_groups:
            changrp_cost = self._estimate_channel_group_cost(changrp_slice)

            # TODO: make this cost estimate more general
            assert len(self._row_col_slices) == self._num_threads
            cost += changrp_cost * max(
                self._estimate_row_slice_cost(row_slice)
                for row_slice in self._row_col_slices
            )

        return cost


class ParallelizationPlanner(ABC):
    def __init__(
        self,
        *,
        num_threads: int,
        forced: bool = False,
        fixed_cost_per_thread: SupportsFloat = 0,
    ) -> None:
        assert 0 < num_threads <= MAX_THREADS
        self.logger = logging.getLogger(self.__class__.__name__)
        self._num_threads = num_threads
        self._forced = forced
        self._fixed_cost_per_thread = fixed_cost_per_thread

    @abstractmethod
    def create_n_thread_candidates(self, num_threads: int) -> None:
        pass

    def create_candidate_plans(self) -> None:
        for n in range(self._num_threads):
            self.create_n_thread_candidates(n + 1)

    @abstractmethod
    def find_optimal_plan(self) -> ParallelizationPlan:
        raise NotImplementedError()


_P = TypeVar("_P", bound=ParallelizationPlan)


class NaiveParallelizationPlanner(ParallelizationPlanner, Generic[_P]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._candidate_plans: List[_P] = []

    def add_candidate_plan(self, plan: _P) -> None:
        self._candidate_plans.append(plan)

    def find_optimal_plan(self) -> _P:
        if not self._candidate_plans:
            self.create_candidate_plans()

        best_plan = min(self._candidate_plans, key=lambda plan: plan.estimate_cost())

        if best_plan._num_threads == self._num_threads:
            self.logger.debug(f"found best plan: {repr(best_plan)}")
            return best_plan
        else:
            forced_candidates = [
                plan
                for plan in self._candidate_plans
                if plan._num_threads == self._num_threads
            ]
            best_forced_plan = None
            if forced_candidates:
                best_forced_plan = min(
                    forced_candidates, key=lambda plan: plan.estimate_cost()
                )

        if self._forced:
            if best_forced_plan:
                self.logger.warning(
                    f"forcing suboptimal plan {repr(best_forced_plan)} "
                    f"when better alternative {repr(best_plan)} exists."
                )
                return best_forced_plan

            self.logger.warning(
                f"no forced plan could be found, resolving to {repr(best_plan)}"
            )
        else:
            self.logger.debug(
                f"replacing suboptimal plan {repr(best_forced_plan)} "
                f"with better alternative {repr(best_plan)}."
            )
        return best_plan


class ElementWisePlanner(NaiveParallelizationPlanner[ElementWiseParallelizationPlan]):
    def __init__(self, num_elements: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert num_elements > 0
        self._num_elements = num_elements

    def create_n_thread_candidates(self, num_threads: int) -> None:
        # TODO: expose the alignment as an option
        r = self._num_elements % VECTOR_SIZE_BYTES
        full_vectors = (self._num_elements - r) // VECTOR_SIZE_BYTES
        p = full_vectors % num_threads
        k = (full_vectors - p) // num_threads

        job_sizes = [
            k * VECTOR_SIZE_BYTES + (idx < p) * VECTOR_SIZE_BYTES
            for idx in range(num_threads)
        ]
        job_sizes[-1] += r
        assert sum(job_sizes) == self._num_elements

        if 0 not in job_sizes:
            self.add_candidate_plan(
                ElementWiseParallelizationPlan(
                    num_threads,
                    job_sizes=job_sizes,
                    fixed_cost_per_thread=self._fixed_cost_per_thread,
                )
            )


class ChannelGroupSlicePlanner(
    NaiveParallelizationPlanner[ChannelGroupParallelizationPlan]
):
    def __init__(self, num_channels_out: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cout = num_channels_out

    def split_channelwise(self) -> List[_ChannelGroup]:
        changrps = []
        num_changrps = math.ceil(self._cout / CHANNEL_GROUP_SIZE)
        for i in range(num_changrps):
            Cbegin = i * CHANNEL_GROUP_SIZE
            Cend = min(Cbegin + CHANNEL_GROUP_SIZE - 1, self._cout - 1)
            changrps.append(_ChannelGroup(Cbegin, Cend))

        return changrps

    def create_n_thread_candidates(self, num_threads: int) -> None:
        changrps = self.split_channelwise()
        if len(changrps) >= num_threads:
            self.add_candidate_plan(
                ChannelGroupParallelizationPlan(
                    num_threads,
                    channel_groups=changrps,
                    fixed_cost_per_thread=self._fixed_cost_per_thread,
                )
            )


class SlicePlanner(NaiveParallelizationPlanner[RowColumnParallelizationPlan]):
    def __init__(
        self, num_channels_out: int, height: int, width: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert height * width > 0, f"received height={height}, width={width}"
        self._height, self._width = height, width
        self._ch_group_planner = ChannelGroupSlicePlanner(num_channels_out, **kwargs)

    def _split_unidirectionally(
        self, dim: int, num_threads: int
    ) -> Tuple[List[int], List[int]]:
        adjustments: Dict[int, Callable[[int], List[int]]] = {
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

    def _split_vertically(self, num_threads: int) -> List[_RowColumnSlice]:
        starts, heights = self._split_unidirectionally(self._height, num_threads)
        return [
            _RowColumnSlice(starts[j], 0, heights[j], self._width)
            for j in range(num_threads)
            if heights[j] > 0
        ]

    def create_n_thread_candidates(self, num_threads: int) -> None:
        self.add_candidate_plan(
            RowColumnParallelizationPlan(
                num_threads,
                channel_groups=self._ch_group_planner.split_channelwise(),
                row_column_slices=self._split_vertically(num_threads),
                fixed_cost_per_thread=self._fixed_cost_per_thread,
            )
        )
