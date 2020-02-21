#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
from tflite2xcore.model_generation.interface import FunctionModel
import op_test_models_common as common

DEFAULT_INPUTS = 16
DEFAULT_WIDTH = 3
DEFAULT_HEIGHT = 5
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'relu6').resolve()


class ReLU6(FunctionModel):
    def build(self, height, width, input_channels, *, input_init):
        class ReLU6Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self._name = 'relu6model'
                self._trainable = False
                self._expects_training_arg = False

            @tf.function
            def func(self, x):
                return tf.nn.relu6(x)

        self.core_model = ReLU6Model()
        self.input_shape = (height, width, input_channels)
        self.input_init = input_init

    @property
    def concrete_function(self):
        return self.core_model.func.get_concrete_function(
            tf.TensorSpec([1, *self.input_shape], tf.float32))

    def prep_data(self):  # Not training this model
        pass

    def train(self):  # Not training this model
        pass

    # TODO: fix this hack
    def gen_test_data(self, *args, **kwargs):
        common.DefaultOpTestModel.gen_test_data(self, *args, **kwargs)


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         input_init=common.DEFAULT_UNIF_INIT):

    # Instantiate model
    test_model = ReLU6('relu6', Path(path))
    test_model.build(height, width, input_channels,
                     input_init=input_init)
    test_model.gen_test_data()
    test_model.save_core_model()
    test_model.populate_converters()


if __name__ == "__main__":
    parser = common.OpTestDimParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         input_init=initializers['input_init'])
