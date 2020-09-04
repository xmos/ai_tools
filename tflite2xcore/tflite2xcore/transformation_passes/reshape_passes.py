# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np  # type: ignore

from tflite2xcore.xcore_schema import BuiltinOpCodes
from tflite2xcore.xcore_model import Operator, Tensor

from .transformation_passes import OperatorMatchingPass


class RemoveFlattenReshapePass(OperatorMatchingPass):
    MATCHING_OPCODES = (
        # TODO fully populate this set e.g. average pooling
        BuiltinOpCodes.FULLY_CONNECTED,
    )

    def match(self, op: Operator) -> bool:
        try:
            producer = op.inputs[0].producers[0]
        except IndexError:
            # Input tensor for op has no producers..
            return False

        # FULLY_CONNECTED always interprets the first dim as batch...
        reshape_input_batch = producer.inputs[0].shape[0]
        reshape_output_batch = op.inputs[0].shape[0]

        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and producer.operator_code.code is BuiltinOpCodes.RESHAPE
            and reshape_output_batch == reshape_input_batch
        )

    def mutate(self, op: Operator) -> None:

        producer = op.inputs[0].producers[0]

        # Remove connection from old inputs to the anchor FC op
        intermediate = op.inputs[0]
        intermediate.consumers.remove(op)

        # Create the new connection
        op.inputs[0] = producer.inputs[0]
        producer.inputs[0].consumers.append(op)


class CanonicalizeReshapePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        if not (super().match(op) and op.operator_code.code is BuiltinOpCodes.RESHAPE):
            return False

        try:
            if op.builtin_options["new_shape"] != list(op.outputs[0].shape):
                self.logger.warning(
                    "new_shape option to RESHAPE doesn't match output tensor shape"
                )
        except (KeyError, TypeError):
            # TODO: consider removing this since in tf2.2 the builtin options seems unused
            self.logger.warning("Expected new_shape option to RESHAPE was not found")

        if -1 in op.inputs[0].shape + op.outputs[0].shape:
            self.logger.warning("Dynamically sized tensors are not supported")
            return False

        assert np.prod(op.inputs[0].shape) == np.prod(
            op.outputs[0].shape
        ), "RESHAPE input and output shapes are not consistent"

        return len(op.inputs) == 2 and op.inputs[1].is_constant

    def mutate(self, op: Operator) -> None:
        # Remove connection between RESHAPE and input tensor[1], the new shape
        op.inputs[1].consumers.remove(op)
        op.inputs = op.inputs[:1]
