#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_lookup_8 import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_INPUTS
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'prelu').resolve()


class PReLU(common.OpTestDefaultModel):
    def build_core_model(self, height, width, input_channels, *,
                         input_init, alpha_init):
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.PReLU(input_shape=(height, width, input_channels),
                                      alpha_initializer=alpha_init)
            ]
        )


def main(raw_args=None):
    parser = common.OpTestImgParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'inits': {
            'input_init': {
                'type': common.OpTestInitializers.UNIF,
                'help': "Initializer for input data distribution."
            },
            'alpha_init': {
                'type': common.OpTestInitializers.UNIF,
                'help': "Initializer for learnable parameters."
            }
        }
    })
    args = parser.parse_args(raw_args)

    test_model = PReLU('prelu', args.path)
    test_model.build(args.height, args.width, args.inputs, **args.inits)
    test_model.run()


if __name__ == "__main__":
    main()
