#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_avgpool2d import (
    DEFAULT_INPUTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)

DEFAULT_PATH = Path(__file__).parent.joinpath("debug", "avgpool2d_global").resolve()


class AvgPool2DGlobal(common.OpTestDefaultModel):
    def build_core_model(self, height, width, input_channels, *, input_init):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.GlobalAveragePooling2D(
                    input_shape=(height, width, input_channels)
                )
            ],
        )


def main(raw_args=None):
    parser = common.OpTestImgParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "inits": {"input_init": {"type": common.OpTestInitializers.UNIF}},
        }
    )
    args = parser.parse_args(raw_args)

    model = AvgPool2DGlobal("avgpool2d_global", args.path)
    model.build(args.height, args.width, args.inputs, **args.inits)
    model.run()


if __name__ == "__main__":
    main()
