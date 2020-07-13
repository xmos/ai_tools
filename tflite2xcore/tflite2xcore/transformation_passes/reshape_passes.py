# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np  # type: ignore

from tflite2xcore.xcore_schema import (
    QuantizationDetails,
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)
from .transformation_passes import OperatorMatchingPass
from tflite2xcore.xcore_model import Operator, Tensor


class RemoveFlattenReshapePass(OperatorMatchingPass):

    MATCHING_OPCODES = (
        # TODO fully populate this set e.g. average pooling
        BuiltinOpCodes.FULLY_CONNECTED,
    )

    @property
    def _producer(self) -> Tensor:
        return self._op.inputs[0].producers[0]

    def match(self, op: Operator) -> bool:

        with self.using(op):
            try:
                producer_opcode = self._producer.operator_code.code
            except IndexError:
                # Input tensor for op has no producers..
                return False

            # FULLY_CONNECTED always interprets the first dim as batch...
            reshape_input_batch = self._producer.inputs[0].shape[0]
            reshape_output_batch = op.inputs[0].shape[0]

            return (
                super().match(op)
                and op.operator_code.code in self.MATCHING_OPCODES
                and producer_opcode is BuiltinOpCodes.RESHAPE
                and (reshape_output_batch == reshape_input_batch)
            )

    def mutate(self, op: Operator) -> None:
        subgraph = op.subgraph

        with self.using(op):
            producer = self._producer

        # Remove connection from old inputs to the anchor FC op
        intermediate = op.inputs[0]
        intermediate.consumers.remove(op)

        # Create the new connection
        op.inputs[0] = producer.inputs[0]
        producer.inputs[0].consumers.append(op)


class CanonicalizeReshapePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        if op.operator_code.code is BuiltinOpCodes.RESHAPE:

            try:
                if op.builtin_options["new_shape"] != list(op.outputs[0].shape):
                    self.logger.warning(
                        "new_shape option to RESHAPE doesn't match output tensor shape"
                    )
            except KeyError:
                self.logger.warning(
                    "Expected new_shape option to RESHAPE was not found"
                )

            assert np.prod(op.inputs[0].shape) == np.prod(
                op.outputs[0].shape
            ), "RESHAPE input and output shapes are not consistent"

            assert -1 not in op.inputs[0].shape and -1 not in op.outputs[0].shape

            return (
                super().match(op) and len(op.inputs) == 2 and op.inputs[1].is_constant
            )
        else:
            return False

    def mutate(self, op: Operator) -> None:
        # Remove connection between RESHAPE and input tensor[1], the new shape
        op.inputs[1].consumers.remove(op)
        op.inputs = op.inputs[:1]
