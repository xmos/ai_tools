# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import Iterable

from tflite2xcore.xcore_model import Operator, Subgraph
from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)

from .transformation_passes import SubgraphAnalysisPass


class ReplaceAddPass(SubgraphAnalysisPass):
    def match(self, op: Operator) -> bool:
        return super().match(op) and op.operator_code is BuiltinOpCodes.ADD

    def target_iterable(self, subgraph: Subgraph) -> Iterable[Operator]:
        return subgraph.operators

    def log_match(self, op: Operator) -> None:
        self.logger.info(f"ADD Operator: {op}")

    def run_subgraph(self, subgraph: Subgraph) -> int:
        super().run_subgraph(subgraph)
        if self._num_matches:
            self.logger.warning(f"ADD Operator Found: {self._num_matches}")
        return 0
