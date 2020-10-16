# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import Iterable

from tflite2xcore.xcore_model import Operator, Subgraph
from tflite2xcore.xcore_schema import BuiltinOpCodes

from .transformation_passes import SubgraphTransformationPass


class FloatingPointWarningPass(SubgraphTransformationPass):
    def mutate(self, obj):
        pass

    def match(self, op: Operator) -> bool:
        return super().match(op) and op is BuiltinOpCodes.ADD

    def target_iterable(self, subgraph: Subgraph) -> Iterable[Operator]:
        return subgraph.operatos

    def log_match(self, op: Operator) -> None:
        self.logger.info(f"ADD Operator: {op}")
