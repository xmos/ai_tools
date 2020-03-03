# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import abstractmethod

from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.graph_transformer import PassPriority
from .transformation_passes import QuantizedOperatorMatchingPass


class ReplacePool2DPass(QuantizedOperatorMatchingPass):
    @property
    def _strides(self):
        options = self._op.builtin_options
        return options['stride_h'], options['stride_w']

    @property
    def _pool_size(self):
        options = self._op.builtin_options
        return options['filter_height'], options['filter_width']

    @property
    def _padding(self):
        return self._op.builtin_options['padding']

    @property
    def _fused_activation(self):
        return self._op.builtin_options['fused_activation_function']

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return ("XC_op" not in op.custom_options
                        and self._input.quantization == self._output.quantization
                        and self._fused_activation == 'NONE'
                        and self._input.shape[3] % 4 == 0)

        return False

    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        with self.using(op):
            op.add_custom_options(XC_op=self.new_opcode.code.value)


class ReplacePool2D2x2Pass(ReplacePool2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._strides == (2, 2)
                        and self._pool_size == (2, 2)
                        and self._input.shape[1] % 2 == 0
                        and self._input.shape[2] % 2 == 0)

        return False


class ReplaceMaxPool2DPass(ReplacePool2DPass):
    def __init__(self, priority=PassPriority.MEDIUM, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        if self.safe_mode:
            self.superseding_passes.append(ReplaceMaxPool2D2x2Pass())

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding == 'VALID'

        return False


class ReplaceMaxPool2D2x2Pass(ReplacePool2D2x2Pass, ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)


class ReplaceAveragePool2DPass(ReplacePool2DPass):
    def __init__(self, priority=PassPriority.MEDIUM, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        if self.safe_mode:
            self.superseding_passes.append(ReplaceAveragePool2D2x2Pass())

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding == 'VALID'

        return False


class ReplaceAveragePool2D2x2Pass(ReplacePool2D2x2Pass, ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d)
