# Copyright (c) 2020, XMOS Ltd, All rights reserved

import _pytest
import larq
import numpy as np
import tensorflow as tf
import larq_compute_engine as lce
from typing import Optional, Tuple, Type, Any, Union, NamedTuple

from tflite2xcore.utils import get_bitpacked_shape, unpack_bits  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import (  # type: ignore # TODO: fix this
    Tensor,
    Buffer,
    ExternalOpCodes,
    TensorType,
    XCOREModel,
)
from tflite2xcore.pass_manager import PassManager  # type: ignore # TODO: fix this
from tflite2xcore.transformation_passes import (  # type: ignore # TODO: fix this
    CanonicalizeLceQuantizedInputPass,
    CanonicalizeLceQuantizedOutputPass,
)
from tflite2xcore.transformation_passes.transformation_passes import (  # type: ignore # TODO: fix this
    OutputTensorMatchingPass,
)
from tflite2xcore.converter import CleanupManager  # type: ignore # TODO: fix this

from tflite2xcore.model_generation import Configuration, TFLiteModel, Hook
from tflite2xcore.model_generation.evaluators import LarqEvaluator
from tflite2xcore.model_generation.runners import Runner
from tflite2xcore.model_generation.converters import KerasModelConverter
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory
from tflite2xcore.model_generation.utils import parse_init_config

from .. import (
    IntegrationTestRunner,
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
        self._config["output_range"] = cfg.pop("output_range", (-3, 3))

        self._config["input_range"] = input_range = cfg.pop("input_range")
        cfg["input_init"] = cfg.pop("input_init", ("RandomUniform", *input_range))

        self._config.update(
            {"bias_init": cfg.pop("bias_init", ("RandomUniform", -1, 1))}
        )
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()

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
            bias_initializer=parse_init_config(*cfg["bias_init"]),
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


class RemoveSingleOutputOperatorPass(OutputTensorMatchingPass):  # type: ignore # TODO: fix this
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


class LarqConverter(KerasModelConverter):
    """ Converts a Larq model to a TFLite model. """

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
        self._model = lce.convert_keras_model(
            self._input_model_hook(),
            inference_input_type=tf.int8,
            inference_output_type=tf.int8,
            experimental_enable_bitpacked_activations=True,
        )

        model_ir = XCOREModel.deserialize(self._model)
        pass_mgr = PassManager(model_ir)

        if self._strip:
            pass_mgr.register_pass(CanonicalizeLceQuantizedInputPass())
            pass_mgr.register_pass(CanonicalizeLceQuantizedOutputPass())
        if self._remove_last_op:
            pass_mgr.register_pass(RemoveSingleOutputOperatorPass())
        pass_mgr.register_passes(CleanupManager())

        pass_mgr.run_passes()

        # TODO: fix this in serialization
        # LCE and builtin interpreter expects an empty buffer in the beginning
        b = Buffer(model_ir)
        model_ir.buffers.remove(b)
        model_ir.buffers = [b, *model_ir.buffers]

        self._model = model_ir.serialize()


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BinarizedTestRunner(IntegrationTestRunner):
    _model_generator: LarqCompositeTestModelGenerator

    class OutputData(NamedTuple):
        reference_quant: np.ndarray
        xcore: np.ndarray

    def __init__(
        self,
        generator: Type[LarqCompositeTestModelGenerator],
        *,
        use_device: bool = False,
    ) -> None:
        super().__init__(generator, use_device=use_device)

        self._lce_converter = self.make_lce_converter()
        self.register_converter(self._lce_converter)

        self._lce_evaluator = LarqEvaluator(
            self, self.get_representative_data, self._lce_converter.get_converted_model
        )
        self.register_evaluator(self._lce_evaluator)

    def get_xcore_evaluation_data(self) -> Union[np.ndarray, tf.Tensor]:
        return self.get_representative_data()

    def make_lce_converter(self) -> LarqConverter:
        return LarqConverter(self, self.get_built_model)

    def make_repr_data_factory(self) -> InputInitializerDataFactory:
        return InputInitializerDataFactory(
            self,
            lambda: get_bitpacked_shape(self._model_generator.input_shape),
            dtype=tf.int32,
        )

    def get_xcore_reference_model(self) -> TFLiteModel:
        return self._lce_converter.get_converted_model()

    def run(self) -> None:
        super().run()
        self._lce_converter.convert()
        self._lce_evaluator.evaluate()

        self.rerun_post_cache()

    def rerun_post_cache(self) -> None:
        super().rerun_post_cache()

        self.outputs = self.OutputData(
            self._lce_evaluator.output_data, self._xcore_evaluator.output_data,
        )
        self.converted_models.update(
            {
                "reference_lce": self._lce_converter._model,
                "xcore": self._xcore_converter._model,
            }
        )
