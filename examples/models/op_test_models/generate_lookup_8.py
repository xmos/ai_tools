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
DEFAULT_PATH = Path(__file__).parent.joinpath("debug").resolve()


class LUTActivation(common.OpTestDefaultModel):
    _ACTIVATION_MAP = {
        "relu": lambda *args, **kwargs: tf.keras.layers.Activation(
            "relu", *args, **kwargs
        ),
        "relu6": lambda *args, **kwargs: tf.keras.layers.Lambda(
            lambda x: tf.nn.relu6(x), *args, **kwargs
        ),
        "logistic": lambda *args, **kwargs: tf.keras.layers.Activation(
            "sigmoid", *args, **kwargs
        ),
        "tanh": lambda *args, **kwargs: tf.keras.layers.Activation(
            "tanh", *args, **kwargs
        ),
    }

    ACTIVATIONS = list(_ACTIVATION_MAP.keys())

    def build_core_model(
        self, height, width, input_channels, *, input_init, activation
    ):
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                "Invalid activation. Expected one of "
                f"{self.ACTIVATIONS}, received {activation}"
            )
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                self._ACTIVATION_MAP[activation](
                    input_shape=(height, width, input_channels)
                )
            ],
        )


class OpTestActivationParser(common.OpTestImgParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-act",
            "--activation",
            default=defaults["choices"][0],
            choices=defaults["choices"],
            help="Chosen activation function to build a test model of.",
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        args.path = args.path.joinpath(args.activation)
        return args


def main(raw_args=None):
    parser = OpTestActivationParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "choices": LUTActivation.ACTIVATIONS,
            "inits": {"input_init": {"type": common.OpTestInitializers.UNIF}},
        }
    )
    args = parser.parse_args(raw_args)

    test_model = LUTActivation(args.activation, args.path)
    test_model.build(
        args.height, args.width, args.inputs, activation=args.activation, **args.inits
    )
    test_model.run()


if __name__ == "__main__":
    main()
