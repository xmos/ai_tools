# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np  # type: ignore

from tflite2xcore.xcore_schema import BuiltinOpCodes
from tflite2xcore.xcore_model import Operator

from .transformation_passes import OperatorMatchingPass


class AdjacentReshapeMatchingPass(OperatorMatchingPass):
    @property
    def MATCHING_OPCODES(self):
        return (BuiltinOpCodes.FULLY_CONNECTED,)

    @property
    def _reshape_op(self):
        return self._op.inputs[0].producers[0]

    def match(self, op: Operator) -> bool:
        if super().match(op) and op.operator_code.code in self.MATCHING_OPCODES:
            try:
                with self.using(op):
                    reshape_op = self._reshape_op
            except IndexError:
                return False

            return (
                reshape_op.operator_code.code is BuiltinOpCodes.RESHAPE
                and reshape_op.inputs[0].shape[0] == reshape_op.outputs[0].shape[0]
            )

        return False


class RemoveSubsequentReshapePass(AdjacentReshapeMatchingPass):
    @property
    def _reshape_op(self):
        return self._op.outputs[0].consumers[0]

    def match(self, op: Operator) -> bool:
        if super().match(op):
            with self.using(op):
                if len(self._reshape_op.inputs[0].consumers) == 1:
                    return True
                self.logger.warning(
                    "Subsequent RESHAPE found with more than 1 consumer"
                )
        return False

    def mutate(self, op: Operator) -> None:
        with self.using(op):
            reshape_op = self._reshape_op

        # Remove connection from old output to the anchor op
        # then create the new connection
        op.outputs[0].producers.remove(op)
        op.outputs[0] = reshape_op.outputs[0]
        op.outputs[0].producers.append(op)

        op.subgraph.remove_operator(reshape_op)


class RemovePrecedingReshapePass(AdjacentReshapeMatchingPass):
    def mutate(self, op: Operator) -> None:
        reshape_op = op.inputs[0].producers[0]

        # Remove connection from old input to the anchor op
        # then create the new connection
        op.inputs[0].consumers.remove(op)
        op.inputs[0] = reshape_op.inputs[0]
        op.inputs[0].consumers.append(op)


class CanonicalizeReshapePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        if not (super().match(op) and op.operator_code.code is BuiltinOpCodes.RESHAPE):
            return False

        try:
            if list(op.builtin_options["new_shape"]) != list(op.outputs[0].shape):
                raise ValueError(
                    "new_shape option to RESHAPE doesn't match output tensor shape"
                )
        except (KeyError, TypeError):
            # in tf2.2 the builtin options seems unused
            self.logger.debug(
                "Expected new_shape option to RESHAPE was not found "
                "(ensure you are running tf2.2 or newer)"
            )

        if -1 in op.inputs[0].shape + op.outputs[0].shape:
            self.logger.warning("Dynamically sized tensors are not supported")
            return False

        assert np.prod(op.inputs[0].shape) == np.prod(
            op.outputs[0].shape
        ), "RESHAPE input and output shapes are not consistent"

        # NOTE: we used to check if op.inputs[1] is constant
        #       However since we neither we or the runtime currently supports
        #       dynamic shapes, this is disabled for now to enable better
        #       conversion of certain models (e.g. mobilenet v1) in tf2.3>
        return len(op.inputs) == 2

    def mutate(self, op: Operator) -> None:
        # Remove connection between RESHAPE and input tensor[1], the new shape
        op.inputs[1].consumers.remove(op)
        op.inputs = op.inputs[:1]
