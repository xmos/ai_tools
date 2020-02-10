#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import op_test_models_common as common


DEFAULT_INPUTS = 3
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_shallowin_deepout_relu').resolve()


class Conv2dShallowinDeepoutRelu(KerasModel):
    def build(self, K_h, K_w, height, width, input_channels, output_channels,
              *, padding, bias_init, weight_init):
        assert input_channels <= 4, "Number of input channels must be at most 4"
        assert K_w <= 8, "Kernel width must be at most 8"
        assert output_channels % 16 == 0, "Number of output channels must be multiple of 16"
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        super().build()

        # Building
        try:
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
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e

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


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS, output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT, K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING,
         bias_init=common.DEFAULT_CONST_INIT, weight_init=common.DEFAULT_UNIF_INIT):
    kwargs = {
        'name': 'conv2d_shallowin_deepout_relu',
        'path': path if path else DEFAULT_PATH
    }
    common.run_main_conv(
        model=Conv2dShallowinDeepoutRelu(**kwargs),
        num_threads=None,
        input_channels=input_channels,
        output_channels=output_channels,
        height=height, width=width,
        K_h=K_h, K_w=K_w, padding=padding,
        bias_init=bias_init, weight_init=weight_init
    )


if __name__ == "__main__":
    parser = common.get_conv_parser(DEFAULT_PATH=DEFAULT_PATH,
                                    DEFAULT_INPUTS=DEFAULT_INPUTS, DEFAULT_OUTPUTS=DEFAULT_OUTPUTS,
                                    DEFAULT_WIDTH=DEFAULT_WIDTH, DEFAULT_HEIGHT=DEFAULT_HEIGHT,
                                    DEFAULT_PADDING=DEFAULT_PADDING,
                                    DEFAULT_KERNEL_HEIGHT=DEFAULT_KERNEL_HEIGHT,
                                    DEFAULT_KERNEL_WIDTH=DEFAULT_KERNEL_WIDTH)
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)
    
    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs, output_channels=args.outputs,
         K_h=args.kernel_height, K_w=args.kernel_width,
         height=args.height, width=args.width,
         padding=args.padding,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'])
