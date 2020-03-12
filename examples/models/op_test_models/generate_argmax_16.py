#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import logging
from pathlib import Path
import numpy as np
import tflite2xcore.converter as xcore_conv
from tflite2xcore import read_flatbuffer, write_flatbuffer, graph_transformer
from tflite2xcore.operator_codes import BuiltinOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 10
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arg_max_16').resolve()


class ArgMax8To16ConversionPass(graph_transformer.OperatorMatchingPass):
    def __init__(self):
        super().__init__(priority=graph_transformer.PassPriority.MEDIUM)

    def match(self, op):
        if op.operator_code.code is BuiltinOpCodes.ARG_MAX:
            return op.inputs[0].type == TensorType.INT8
        return False

    def mutate(self, op):
        tensor = op.inputs[0]
        tensor.type = TensorType.INT16
        tensor.name = f"{op.name}/input"
        tensor.quantization = {
            'min': tensor.quantization['min'],
            'max': tensor.quantization['max'],
            'scale': [tensor.quantization['scale'][0] / 2**8],
            'zero_point': [int(tensor.quantization['zero_point'][0] * 2**8)],
            'details_type': "CustomQuantization",
            'quantized_dimension': 0
        }


class ArgMax16(common.OpTestDefaultModel):
    def build_core_model(self, input_dim):
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Lambda(
                    lambda x: tf.math.argmax(x, axis=1, output_type=tf.int32),
                    input_shape=(input_dim,)
                )
            ]
        )

    @property
    def input_dim(self):
        return self.input_shape[0]

    def gen_test_data(self):
        utils.set_all_seeds()
        x_test_float = np.float32(
            np.random.uniform(0, 1, size=(self.input_dim, self.input_dim)))
        x_test_float += np.eye(self.input_dim)
        self.data['export_data'] = x_test_float
        self.data['quant'] = x_test_float

    def to_tf_stripped(self):
        model = read_flatbuffer(str(self.models['model_quant']))
        xcore_conv.strip_model(model)

        pass_mgr = graph_transformer.PassManager(
            model, passes=[ArgMax8To16ConversionPass()])
        pass_mgr.run_passes()

        self.models['model_stripped'] = self.models['models_dir'] / "model_stripped.tflite"
        write_flatbuffer(model, str(self.models['model_stripped']))

        self._save_visualization('model_stripped')

    def to_tf_xcore(self):
        # super().to_tf_xcore() converts model_quant
        # to avoid code duplication, here we convert model_stripped instead
        # (because model_stripped has INT16 input that can be matched)
        tmp = self.models['model_quant']
        self.models['model_quant'] = self.models['model_stripped']
        super().to_tf_xcore()
        self.models['model_quant'] = tmp

    def save_tf_stripped_data(self):
        assert 'model_quant' in self.models
        assert 'model_stripped' in self.models

        # load stripped model for quantization info
        model_stripped = read_flatbuffer(str(self.models['model_stripped']))
        input_quant = model_stripped.subgraphs[0].inputs[0].quantization

        # load quant model for inference, b/c the interpreter cannot handle int16 tensors
        interpreter = tf.lite.Interpreter(
            model_path=str(self.models['model_quant']))

        logging.info(f"Extracting examples for model_stripped...")
        x_test = utils.quantize(self.data['export_data'],
                                input_quant['scale'][0],
                                input_quant['zero_point'][0],
                                dtype=np.int16)
        y_test = utils.apply_interpreter_to_examples(interpreter,
                                                     self.data['export_data'])
        data = {'x_test': x_test, 'y_test': np.vstack(list(y_test))}

        self._save_data_dict(data, base_file_name='model_stripped')

    def save_tf_xcore_data(self):
        assert 'model_xcore' in self.models

        model = read_flatbuffer(str(self.models['model_xcore']))
        input_quant = model.subgraphs[0].inputs[0].quantization

        # quantize test data
        x_test = utils.quantize(self.data['export_data'],
                                input_quant['scale'][0],
                                input_quant['zero_point'],
                                dtype=np.int16)

        # save data
        self._save_data_dict({'x_test': x_test}, base_file_name='model_xcore')


def main(raw_args=None):
    parser = common.DefaultParser(defaults={
        'path': DEFAULT_PATH,
    })
    parser.add_argument(
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Number of input channels')
    args = parser.parse_args(raw_args)
    utils.set_gpu_usage(False, args.verbose)

    test_model = ArgMax16('arg_max_16', args.path)
    test_model.build(args.inputs)
    test_model.run()


if __name__ == "__main__":
    main()
