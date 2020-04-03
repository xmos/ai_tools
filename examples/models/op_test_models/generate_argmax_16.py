#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

from pathlib import Path
import numpy as np
from tflite2xcore.serialization import serialize_model, deserialize_model
from tflite2xcore.transformation_passes import OperatorMatchingPass
from tflite2xcore.pass_manager import PassManager
from tflite2xcore.operator_codes import BuiltinOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.utils import set_all_seeds
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 10
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arg_max_16').resolve()


class ArgMax8To16ConversionPass(OperatorMatchingPass):
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
        set_all_seeds()
        x_test_float = np.float32(
            np.random.uniform(0, 1, size=(self.input_dim, self.input_dim)))
        x_test_float += np.eye(self.input_dim)
        self.data['export'] = self.data['quant'] = x_test_float

    def convert_to_stripped(self, **converter_args):
        super().convert_to_stripped(**converter_args)
        model = deserialize_model(self.buffers['model_stripped'])

        pass_mgr = PassManager(
            model, passes=[ArgMax8To16ConversionPass()])
        pass_mgr.run_passes()

        self.buffers['model_stripped'] = serialize_model(model)

    def convert_to_xcore(self, **converter_args):
        super().convert_to_xcore(source='model_stripped', **converter_args)

    def save_stripped_data(self):
        assert 'model_quant' in self.buffers
        assert 'model_stripped' in self.buffers

        # load stripped model for quantization info
        model_stripped = deserialize_model(self.buffers['model_stripped'])
        input_quant = model_stripped.subgraphs[0].inputs[0].quantization

        # load quant model for inference, b/c the interpreter cannot handle int16 tensors
        interpreter = tf.lite.Interpreter(model_content=self.buffers['model_quant'])

        self.logger.debug("Extracting and saving examples for model_stripped...")
        x_test = utils.quantize(self.data['export'],
                                input_quant['scale'][0],
                                input_quant['zero_point'][0],
                                dtype=np.int16)
        y_test = utils.apply_interpreter_to_examples(interpreter,
                                                     self.data['export'])
        data = {'x_test': x_test, 'y_test': y_test}

        self._save_data_dict(data, base_file_name='model_stripped')

    def save_xcore_data(self):
        assert 'model_xcore' in self.buffers

        model = deserialize_model(self.buffers['model_xcore'])
        input_quant = model.subgraphs[0].inputs[0].quantization

        # quantize test data
        x_test = utils.quantize(self.data['export'],
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

    test_model = ArgMax16('arg_max_16', args.path)
    test_model.build(args.inputs)
    test_model.run()


if __name__ == "__main__":
    main()
