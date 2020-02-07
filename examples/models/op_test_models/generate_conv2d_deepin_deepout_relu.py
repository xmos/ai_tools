#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
import logging
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 32
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_deepin_deepout_relu').resolve()
DEFAULT_NUM_THREADS = 1


class Conv2dDeepinDeepoutRelu(KerasModel):
    def build(self, K_h, K_w, height, width, input_channels, output_channels,
              *, padding, bias_init, weight_init):
        assert input_channels % 32 == 0, "# of input channels must be multiple of 32"
        assert output_channels % 16 == 0, "# of output channels must be multiple of 16"
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        super().build()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Conv2D(filters=output_channels,
                                       kernel_size=(K_h, K_w),
                                       padding=padding,
                                       input_shape=(height, width, input_channels),
                                       bias_initializer=bias_init,
                                       kernel_initializer=weight_init)
            ]
        )
        # Compilation
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    def train(self):  # Not training this model
        pass

    # For training
    def prep_data(self, height, width):
        self.data['export_data'], self.data['quant'] = utils.generate_dummy_data(*self.input_shape)

    # For exports
    def gen_test_data(self, height, width):
        if not self.data:
            self.prep_data(height, width)


def main(path=DEFAULT_PATH, *, num_threads=DEFAULT_NUM_THREADS,
         input_channels=DEFAULT_INPUTS, output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT, K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING,
         bias_init=common.DEFAULT_CONST_INIT, weight_init=common.DEFAULT_UNIF_INIT):

    # Instantiate model
    test_model = Conv2dDeepinDeepoutRelu('conv2d_deepin_deepout_relu', Path(path))
    # Build model and compile
    test_model.build(K_h, K_w, height, width, input_channels, output_channels,
                     padding=padding, bias_init=bias_init, weight_init=weight_init)
    # Generate test data
    test_model.gen_test_data(height, width)
    # Save model
    test_model.save_core_model()
    # Populate converters
    test_model.populate_converters(xcore_num_threads=num_threads)


if __name__ == "__main__":
    parser = common.get_conv_parser(DEFAULT_INPUTS=DEFAULT_INPUTS, DEFAULT_OUTPUTS=DEFAULT_OUTPUTS,
                            DEFAULT_WIDTH=DEFAULT_WIDTH, DEFAULT_HEIGHT=DEFAULT_HEIGHT,
                            DEFAULT_PADDING=DEFAULT_PADDING,
                            DEFAULT_KERNEL_HEIGHT=DEFAULT_KERNEL_HEIGHT,
                            DEFAULT_KERNEL_WIDTH=DEFAULT_KERNEL_WIDTH)
    parser.add_argument(
        '-par', '--par_num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         num_threads=args.par_num_threads,
         input_channels=args.inputs, output_channels=args.outputs,
         K_h=args.kernel_height, K_w=args.kernel_width,
         height=args.height, width=args.width,
         padding=args.padding,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'])
