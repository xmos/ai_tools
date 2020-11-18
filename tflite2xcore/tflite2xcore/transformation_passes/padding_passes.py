# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    BuiltinOpCodes,
    XCOREOpCodes,
    OperatorCode,
    Operator,
)

from .transformation_passes import OperatorMatchingPass


class FuseConv2dPaddingPass(OperatorMatchingPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_depthwise,
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_shallowin,
    )

    @property
    def _producer(self):
        return self._op.inputs[0].producers[0]

    @property
    def _pad_params(self):
        return self._producer.inputs[1].as_array().tolist()

    @property
    def _kernel_size(self):
        opcode = self._op.operator_code.code
        weights = self._op.inputs[1]
        if opcode is XCOREOpCodes.XC_conv2d_depthwise:
            return weights.shape[0:2]
        elif opcode in (XCOREOpCodes.XC_conv2d_deep, XCOREOpCodes.XC_conv2d_shallowin):
            return weights.shape[1:3]

    @staticmethod
    def _calculate_end_padding(out_size, strides, in_size, kernel_size):
        return tuple(
            (o - 1) * s - i + k
            for o, s, i, k in zip(out_size, strides, in_size, kernel_size)
        )

    def match(self, op):
        if not super().match(op):
            return False

        with self.using(op):
            opcode = self._op.operator_code.code
            if opcode not in self.MATCHING_OPCODES:
                return False

            try:
                if self._producer.operator_code.code is not BuiltinOpCodes.PAD:
                    return False
            except IndexError:
                # No producers found for input
                return False

            pad_params = self._pad_params
            if len(pad_params) != 4:
                return False

            if pad_params[0] != [0, 0] or pad_params[3] != [0, 0]:
                # NOTE: SplitPaddingPass decouples channel- and batch-wise padding
                return False

            kernel_size = self._kernel_size
            implicit_end_pads = self._calculate_end_padding(
                out_size=op.outputs[0].shape[1:3],
                strides=op.custom_options["stride"],
                in_size=op.inputs[0].shape[1:3],
                kernel_size=kernel_size,
            )

        pad = op.custom_options["pad"]
        all_pads = (
            [pad[0] + pad_params[1][0], implicit_end_pads[0] + pad_params[1][1]],
            [pad[1] + pad_params[2][0], implicit_end_pads[1] + pad_params[2][1]],
        )

        for p, k in zip(all_pads, kernel_size):
            if p[0] >= k or p[1] >= k:
                # kernels currently don't support this
                self.logger.warning(
                    f"While fusing, found implicit padding={p}"
                    f" not smaller than kernel={kernel_size}"
                )
                return False

        if len(pad) == 3 and not isinstance(pad, str):
            return True
        elif pad in ["SAME", "VALID"] + list(Padding):
            raise ValueError(f"Deprecated 'pad' option in {opcode}: 'pad'={pad}")
        else:
            self.logger.warning(f"Invalid option in {opcode}: 'pad'={pad}")

        return False

    def mutate(self, op):
        with self.using(op):
            producer = self._producer
            pad_params = self._pad_params
            old_pad = op.custom_options["pad"]

        # cut connection to old input
        op.inputs[0].consumers.remove(op)

        # add connection from unpadded input to convolution operator
        op.inputs[0] = producer.inputs[0]
        op.inputs[0].consumers.append(op)

        # set padding: [top, left, zero_point]
        op.custom_options["pad"] = [
            old_pad[0] + pad_params[1][0],
            old_pad[1] + pad_params[2][0],
            old_pad[2],
        ]


class SplitPaddingPass(OperatorMatchingPass):
    @property
    def _pad_params(self):
        return self._op.inputs[1].as_array().tolist()

    def match(self, op):
        if not super().match(op):
            return False

        with self.using(op):
            opcode = self._op.operator_code.code
            if opcode is not BuiltinOpCodes.PAD:
                return False

            pad_params = self._pad_params
            if len(pad_params) != 4:
                return False

            return (pad_params[0] != [0, 0] or pad_params[3] != [0, 0]) and (
                pad_params[1] != [0, 0] or pad_params[2] != [0, 0]
            )

    def mutate(self, op):
        subgraph = op.subgraph

        with self.using(op):
            pad_params = self._pad_params
            pads_NC = [pad_params[0], [0, 0], [0, 0], pad_params[3]]
            pads_HW = [[0, 0], pad_params[1], pad_params[2], [0, 0]]

        # cut connection from old input to the op
        old_input = op.inputs[0]
        old_input.consumers.remove(op)

        # create new parameter tensor for the op, and replace old
        # the old op will become the spatial padding
        # this is needed because multiple ops can share the same parameter tensor
        # NOTE: the old paddings tensor might be dangling and will be cleaned up later
        op.inputs[1].consumers.remove(op)
        op.inputs[1] = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=[4, 2], consumers=[op]
        )
        op.inputs[1].buffer.data = np.int32(pads_HW)

        # create new (batch/channel-wise) operator
        new_op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.PAD), inputs=[old_input],
        )
        subgraph.insert_operator(op, new_op)

        # assign padding tensor to new op
        new_op.inputs.append(
            subgraph.create_tensor(
                f"{new_op.name}/paddings",
                TensorType.INT32,
                shape=[4, 2],
                consumers=[new_op],
            )
        )
        new_op.inputs[1].buffer.data = np.int32(pads_NC)

        # create intermediate tensor and wire it up
        intermediate_shape = [
            size + pad[0] + pad[1] for size, pad in zip(old_input.shape, pads_NC)
        ]
        op.inputs[0] = subgraph.create_tensor(
            f"{new_op.name}/output",
            old_input.type,
            intermediate_shape,
            consumers=[op],
            producers=[new_op],
            quantization=old_input.quantization,
        )
        new_op.outputs.append(op.inputs[0])


class FuseConsecutivePadsPass(OperatorMatchingPass):
    @property
    def _producer(self):
        return self._op.inputs[0].producers[0]

    @property
    def _pad_params(self):
        return self._op.inputs[1].as_array()

    def match(self, op):
        # the anchor is the second of two consecutive PAD ops
        try:
            with self.using(op):
                return (
                    super().match(op)
                    and self._op.operator_code.code is BuiltinOpCodes.PAD
                    and self._producer.operator_code.code is BuiltinOpCodes.PAD
                )
        except IndexError:
            # No producers found for input
            return False

    def mutate(self, op):
        subgraph = op.subgraph
        with self.using(op):
            producer = self._producer
            this_params = self._pad_params
        with self.using(producer):
            producer_params = self._pad_params
        new_params = this_params + producer_params

        # cut connection from old inputs to the anchor op
        intermediate = op.inputs[0]
        intermediate.consumers.remove(op)
        op.inputs[1].consumers.remove(op)

        # create new parameter tensor for the op, and replace old
        # this is needed because multiple ops can share the same parameter tensor
        # NOTE: the old paddings tensor might be dangling and will be cleaned up later
        op.inputs[1] = subgraph.create_tensor(
            f"{op.name}/paddings",
            TensorType.INT32,
            shape=new_params.shape,
            consumers=[op],
        )
        op.inputs[1].buffer.data = new_params.astype(np.int32)

        # set up bypass connection
        op.inputs[0] = producer.inputs[0]
        producer.inputs[0].consumers.append(op)


class RemovePaddingInputPass(OperatorMatchingPass):
    def match(self, op):
        if op.operator_code.code is BuiltinOpCodes.PAD:
            padding = op.inputs[1].as_array().tolist()
            return (
                super().match(op)
                # Match padding only where it is the first operator in the subgraph
                and op.inputs[0] in op.subgraph.inputs
                # Make sure no other op uses this input
                and len(op.inputs[0].consumers) == 1
                # Match only padding in channel direction i.e. inserted for VPU alignment
                and len(padding) == 4
                and padding[-1] != [0, 0]
                and all(pad == [0, 0] for pad in padding[:-1])
            )
        else:
            return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.append(op.outputs[0])
        subgraph.remove_tensor(op.inputs[0])  # DCE doesn't clean up subgraph inputs
        subgraph.remove_operator(op)


class ReplacePadPass(OperatorMatchingPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_pad)

    def match(self, op: Operator) -> bool:
        if super().match and op.operator_code.code is BuiltinOpCodes.PAD:
            padding = op.inputs[1].as_array().tolist()

            # match spatial pad only
            if len(padding) == 4 and padding[-1] == [0, 0] and padding[0] == [0, 0]:
                bytes_per_pixel = op.inputs[0].type.sizeof() * op.inputs[0].shape[3]
                return bytes_per_pixel % 4 == 0

        return False

    def mutate(self, op: Operator) -> None:
        raise NotImplementedError()
