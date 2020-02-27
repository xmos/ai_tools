#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_avgpool2d import (
    DEFAULT_INPUTS, DEFAULT_HEIGHT, DEFAULT_WIDTH,
)

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'avgpool2d_global').resolve()


class AvgPool2DGlobal(common.DefaultOpTestModel):
    def build_core_model(self, height, width, input_channels, *, input_init):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.GlobalAveragePooling2D(
                    input_shape=(height, width, input_channels)
                )
            ]
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         input_init=common.DEFAULT_UNIF_INIT):
    model = AvgPool2DGlobal('avgpool2d_global', Path(path))
    model.build(height, width, input_channels,
                input_init=input_init)
    model.run()


if __name__ == "__main__":
    parser = common.OpTestImgParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        'inits': {'input_init': common.OpTestInitializers.UNIF}
    })
    args = parser.parse_args()
    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         input_init=args.inits['input_init'])
