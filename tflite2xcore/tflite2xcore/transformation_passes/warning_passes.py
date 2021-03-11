# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from typing import Iterable

from tflite2xcore.xcore_model import Tensor, Subgraph
from tflite2xcore.xcore_schema import TensorType

from .transformation_passes import SubgraphAnalysisPass


class FloatingPointWarningPass(SubgraphAnalysisPass):
    def match(self, tensor: Tensor) -> bool:
        return super().match(tensor) and tensor.type in (
            TensorType.FLOAT64,
            TensorType.FLOAT32,
            TensorType.FLOAT16,
        )

    def target_iterable(self, subgraph: Subgraph) -> Iterable[Tensor]:
        return subgraph.tensors

    def log_match(self, tensor: Tensor) -> None:
        self.logger.info(f"Floating Point Tensor: {tensor}")

    def run_subgraph(self, subgraph: Subgraph) -> int:
        super().run_subgraph(subgraph)
        if self._num_matches:
            self.logger.warning(f"Floating Point Tensors Found: {self._num_matches}")
        return 0
