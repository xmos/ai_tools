# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np
import tensorflow as tf
from copy import deepcopy
from typing import Iterable

from tflite2xcore.xcore_schema import BuiltinOpCodes
from tflite2xcore.xcore_model import XCOREModel, Operator, Subgraph, Tensor

from .transformation_passes import OperatorMatchingPass


class ConstantPropagationPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        if super().match(op):
            for t in op.inputs:
                if not t.is_constant:
                    return False
                elif not t.buffer.data:
                    self.logger.warning("Found constant tensor with empty buffer")

            if op.operator_code.code in BuiltinOpCodes:
                return True
            else:
                self.logger.warning(
                    f"Found unsupported operator {op.operator_code.code}"
                )

        return False

    def mutate(self, op: Operator) -> None:
        # we first clone a single op model from the op
        new_model = XCOREModel()
        new_subgraph = new_model.create_subgraph()

        def clone_tensors(old_tensors: Iterable[Tensor]) -> Iterable[Tensor]:
            return (new_subgraph.clone_tensor(t) for t in old_tensors)

        new_op = new_subgraph.create_operator(
            op.operator_code,
            inputs=clone_tensors(op.inputs),
            outputs=clone_tensors(op.outputs),
            builtin_options=deepcopy(op.builtin_options),
        )

        # the new model will have no inputs (all op inputs are constant)
        # all op outputs will be subgraph outputs as well
        for tensor in new_op.outputs:
            new_subgraph.outputs.append(tensor)
            # rearranging buffers to satisfy the builtin interpreter
            new_model.buffers.remove(tensor.buffer)
            new_model.buffers = [
                tensor.buffer,
                *new_model.buffers,
            ]

        # run the single op model thourhg the builtin interpreter
        # to get the propagated values
        self.logger.debug("Propagating constant using tf.lite.Interpreter...")
        interp = tf.lite.Interpreter(model_content=new_model.serialize())
        interp.allocate_tensors()
        interp.invoke()
        output_values = [
            interp.get_tensor(det["index"]) for det in interp.get_output_details()
        ]

        # finally, mutate the original graph
        assert len(op.outputs) == len(output_values)  # sanity check
        for tensor, data in zip(op.outputs, output_values):
            tensor.buffer.owners.remove(tensor)
            tensor.buffer = op.model.create_buffer(np.array(data))
            tensor.buffer.owners.append(tensor)
        op.subgraph.remove_operator(op)
