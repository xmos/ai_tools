# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import examples_common as common
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import model_interface as mi
import tflite_utils
import graph_transformer
import tflite2xcore_conv as xcore_conv
from tflite2xcore import read_flatbuffer, write_flatbuffer
from operator_codes import BuiltinOpCodes
from xcore_model import TensorType

DEFAULT_INPUTS = 10
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arg_max_16').resolve()


class ArgMax8To16ConversionPass(graph_transformer.OperatorMatchingPass):
    def __init__(self):
        super().__init__(priority=graph_transformer.PassPriority.MEDIUM)

    def match(self, op):
        if op.operator_code.code.value == BuiltinOpCodes.ARG_MAX.value:  # TODO: fix this
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


class ArgMax16(mi.FunctionModel):
    def build(self, input_dim):
        class ArgMaxModel(tf.keras.Model):

            def __init__(self):
                super(ArgMaxModel, self).__init__()
                self._name = 'argmaxmodel'
                self._trainable = False
                self._expects_training_arg = False
                pass

            @tf.function
            def func(self, x):
                return tf.math.argmax(x, axis=1, output_type=tf.int32)
        self.core_model = ArgMaxModel()
        self.input_dim = input_dim

    @property
    def concrete_function(self):
        return self.core_model.func.get_concrete_function(
            tf.TensorSpec([1, self.input_dim], tf.float32)
        )

    def prep_data(self):  # Not training this model
        pass

    def train(self):  # Not training this model
        pass

    def gen_test_data(self):
        tflite_utils.set_all_seeds()
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

    def save_tf_stripped_data(self):
        assert 'model_quant' in self.models
        assert 'model_stripped' in self.models

        # load stripped model for quantization info
        model_stripped = read_flatbuffer(str(self.models['model_stripped']))
        input_quant = model_stripped.subgraphs[0].inputs[0].quantization

        # load quant model for inference, b/c the interpreter cannot handle int16 tensors
        interpreter = tf.lite.Interpreter(model_path=str(self.models['model_quant']))

        base_file_name = 'model_stripped'
        logging.info(f"Extracting examples for {base_file_name}...")
        x_test = common.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'][0], dtype=np.int16)
        y_test = common.apply_interpreter_to_examples(interpreter, self.data['export_data'])
        data = {'x_test': x_test,
                'y_test': np.vstack(list(y_test))}
        common.save_test_data(data, self.models['data_dir'], base_file_name)

    def save_tf_xcore_data(self):
        assert 'model_xcore' in self.models

        model = read_flatbuffer(str(self.models['model_xcore']))
        input_quant = model.subgraphs[0].inputs[0].quantization

        # quantize test data
        x_test = common.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'], dtype=np.int16)

        # save data
        common.save_test_data({'x_test': x_test}, self.models['data_dir'], 'model_xcore')


def main(path=DEFAULT_PATH, input_dim=DEFAULT_INPUTS):
    # Instantiate model
    test_model = ArgMax16('arg_max_16', Path(path))

    # Build model
    test_model.build(input_dim)

    # Export data generation
    test_model.gen_test_data()

    # Save model
    test_model.save_core_model()

    # Populate converters and data
    test_model.populate_converters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Input dimension')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    args = parser.parse_args()

    # TODO: consider refactoring this to utils
    verbose = args.verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    main(path=args.path, input_dim=args.inputs)
