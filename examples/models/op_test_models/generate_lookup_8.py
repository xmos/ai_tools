#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 16
DEFAULT_WIDTH = 3
DEFAULT_HEIGHT = 5
DEFAULT_PATH = Path(__file__).parent.joinpath('debug').resolve()


class LUTActivation(common.DefaultOpTestModel):
    _ACTIVATION_MAP = {
        'relu': lambda *args, **kwargs:
            tf.keras.layers.Activation('relu', *args, **kwargs),
        'relu6': lambda *args, **kwargs:
            tf.keras.layers.Lambda(lambda x: tf.nn.relu6(x), *args, **kwargs),
        'logistic': lambda *args, **kwargs:
            tf.keras.layers.Activation('sigmoid', *args, **kwargs),
        'tanh': lambda *args, **kwargs:
            tf.keras.layers.Activation('tanh', *args, **kwargs)
    }

    ACTIVATIONS = list(_ACTIVATION_MAP.keys())

    def build_core_model(self, height, width, input_channels, *,
                         input_init, activation):
        if activation not in self._ACTIVATION_MAP:
            raise ValueError("Invalid activation. Expected one of "
                             f"{self.ACTIVATIONS}, received {activation}")
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                self._ACTIVATION_MAP[activation](input_shape=(height, width, input_channels))
            ]
        )


def main(activation, path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         input_init=common.DEFAULT_UNIF_INIT):
    test_model = LUTActivation(activation, Path(path))
    test_model.build(height, width, input_channels,
                     input_init=input_init, activation=activation)
    test_model.run()


class OpTestActivationParser(common.OpTestImgParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '-act', '--activation', default=defaults['choices'][0],
            choices=defaults['choices'],
            help='Chosen activation function to build a test model of.'
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        print(args)
        args.path = Path(args.path).joinpath(args.activation)
        return args


if __name__ == "__main__":
    parser = OpTestActivationParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "choices": LUTActivation.ACTIVATIONS
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = parser.initializer_args_handler(args)

    main(args.activation,
         path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         input_init=initializers['input_init'])
