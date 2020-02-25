#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_lookup_8 import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_INPUTS
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'prelu').resolve()


class PReLU(common.DefaultOpTestModel):
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


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         alpha_init=common.DEFAULT_UNIF_INIT,
         input_init=common.DEFAULT_UNIF_INIT):
    test_model = PReLU('prelu', Path(path))
    test_model.build(height, width, input_channels,
                     input_init=input_init, alpha_init=alpha_init)
    test_model.run()


class OpTestPReLUParser(common.OpTestImgParser):
    def add_initializers(self):
        super().add_initializers()
        self.add_argument(
            "--alpha_init", nargs="*", default=argparse.SUPPRESS,
            help="Initialize learnable parameters. Possible initializers are: "
                 "const [CONST_VAL] or unif [MIN MAX]. "
                 f"(default: {common.OpTestInitializers.UNIF.value} "
                 f"{common._DEFAULT_UNIF_INIT[0]} {common._DEFAULT_UNIF_INIT[1]})",
        )


if __name__ == "__main__":
    parser = OpTestPReLUParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = parser.initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         alpha_init=initializers['alpha_init'],
         input_init=initializers['input_init'])
