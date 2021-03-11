# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import _pytest
import larq
import numpy as np
import tensorflow as tf
import larq_compute_engine as lce
from typing import Optional, Tuple, Type, Any, Union, NamedTuple

from tflite2xcore.utils import get_bitpacked_shape
from tflite2xcore.xcore_schema import (
    Tensor,
    Buffer,
    ExternalOpCodes,
    TensorType,
    XCOREModel,
)
from tflite2xcore.pass_manager import PassManager
from tflite2xcore.transformation_passes import (
    CanonicalizeLceQuantizedInputPass,
    CanonicalizeLceQuantizedOutputPass,
    UnifyEmptyBuffersPass,
)
from tflite2xcore.transformation_passes.transformation_passes import (
    OutputTensorMatchingPass,
)
from tflite2xcore.converter import CleanupManager

from tflite2xcore.model_generation import Configuration, TFLiteModel, Hook
from tflite2xcore.model_generation.evaluators import LarqEvaluator
from tflite2xcore.model_generation.runners import Runner
from tflite2xcore.model_generation.converters import KerasModelConverter, LarqConverter
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory
from tflite2xcore.model_generation.utils import parse_init_config

from .. import (
    BinarizedTestRunner,
    test_reference_model_regression,
    test_converted_single_op_model,
    test_mean_abs_diffs,
    test_output,
)
from ..test_conv2d import Conv2dWordAlignedTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class LarqCompositeTestModelGenerator(Conv2dWordAlignedTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["activation"] = cfg.pop("activation", "linear")

        self._config["output_range"] = cfg.pop("output_range", (-3, 3))

        self._config["input_range"] = input_range = cfg.pop("input_range")
        cfg["input_init"] = cfg.pop("input_init", ("RandomUniform", *input_range))

        self._config.update(
            {"bias_init": cfg.pop("bias_init", ("RandomUniform", -1, 1))}
        )
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()

        assert self._config["activation"] in ("linear", "relu", "relu6")

        input_range = self._config["input_range"]
        assert len(input_range) == 2
        assert input_range[0] <= 0 <= input_range[1]

        output_range = self._config["output_range"]
        assert len(output_range) == 2
        assert output_range[0] <= 0 <= output_range[1]

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Conv2D:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        cfg = self._config
        return larq.layers.QuantConv2D(
            filters=cfg["output_channels"],
            kernel_size=(cfg["K_h"], cfg["K_w"]),
            padding=cfg["padding"],
            pad_values=1,
            strides=cfg["strides"],
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
            kernel_initializer=parse_init_config(*cfg["weight_init"]),
            **kwargs,
        )

    def _fake_quant(
        self, x: tf.Tensor, range_min: int = 0, range_max: int = 1
    ) -> tf.Tensor:
        return tf.quantization.fake_quant_with_min_max_vars(x, range_min, range_max)

    def _build_core_model(self) -> tf.keras.Model:
        img = tf.keras.layers.Input(shape=self._input_shape)
        x = self._fake_quant(img, *self._config["input_range"])
        x = self._op_layer()(x)
        if self._config["activation"] == "relu":
            x = tf.keras.layers.ReLU()(x)
        elif self._config["activation"] == "relu6":
            x = tf.keras.layers.ReLU(6)(x)
        x = tf.keras.layers.BatchNormalization(
            beta_initializer=parse_init_config(*self._config["bias_init"])
        )(x)
        x = self._fake_quant(x, *self._config["output_range"])
        return tf.keras.Model(img, x)


class BConv2dGenericTestModelGenerator(LarqCompositeTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert (
            "input_range" not in cfg
        ), f"input_range cannot be specified for BConv2d tests"
        cfg["input_range"] = (np.iinfo(np.int32).min, np.iinfo(np.int32).max)

        assert (
            "input_init" not in cfg
        ), f"input_init cannot be specified for BConv2d tests"
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 32 == 0
        ), "# of input channels must be multiple of 32"


#  ----------------------------------------------------------------------------
#                                   CONVERTERS
#  ----------------------------------------------------------------------------


class RemoveSingleOutputOperatorPass(OutputTensorMatchingPass):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._done = False

    def match(self, tensor: Tensor) -> bool:
        return (not self._done) and super().match(tensor)

    def mutate(self, tensor: Tensor) -> None:
        subgraph = tensor.subgraph
        out_op = tensor.producers[0]

        subgraph.outputs.remove(tensor)
        subgraph.outputs.append(out_op.inputs[0])
        subgraph.remove_operator(out_op)

        self._done = True


class LarqSingleOpConverter(LarqConverter):
    """ Converts a larq composite TFL model to a single op TFL model.
    
        This converter is to work around the fact that some larq ops
        cannot be directly generated from keras layers.
    """

    def __init__(
        self,
        runner: Runner,
        input_model_hook: Hook[tf.keras.Model],
        strip: bool = False,
        remove_last_op: bool = False,
    ) -> None:
        super().__init__(runner, input_model_hook)
        self._strip = strip
        self._remove_last_op = remove_last_op

    def convert(self) -> None:
        super().convert()

        model_ir = XCOREModel.deserialize(self._model)
        pass_mgr = PassManager(model_ir)

        if self._strip:
            pass_mgr.register_pass(CanonicalizeLceQuantizedInputPass())
            pass_mgr.register_pass(CanonicalizeLceQuantizedOutputPass())
        if self._remove_last_op:
            pass_mgr.register_pass(RemoveSingleOutputOperatorPass())

        pass_mgr.register_pass(UnifyEmptyBuffersPass())
        pass_mgr.register_passes(CleanupManager())

        pass_mgr.run_passes()

        self._model = model_ir.serialize()


#  ----------------------------------------------------------------------------
#                                   CONVERTERS
#  ----------------------------------------------------------------------------


class BinarizedSingleOpRunner(BinarizedTestRunner):
    def make_repr_data_factory(self) -> InputInitializerDataFactory:
        return InputInitializerDataFactory(
            self,
            lambda: get_bitpacked_shape(self._model_generator.input_shape),
            dtype=tf.int32,
        )
