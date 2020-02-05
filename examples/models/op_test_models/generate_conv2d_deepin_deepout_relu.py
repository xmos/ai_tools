#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
import logging
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf

DEFAULT_INPUTS = 32
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
_DEFAULT_CONST_INIT = 0
_DEFAULT_UNIF_INIT = [-1, 1]
DEFAULT_CONST_INIT = tf.constant_initializer(_DEFAULT_CONST_INIT)
DEFAULT_UNIF_INIT = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT)
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_deepin_deepout_relu').resolve()


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


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS, output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT, K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING,
         bias_init=DEFAULT_CONST_INIT, weight_init=DEFAULT_UNIF_INIT):

    # Instantiate model
    test_model = Conv2dDeepinDeepoutRelu('conv2d_deepin_deepout_relu', Path(path))
    # Build model and compile
    test_model.build(K_h, K_w, height, width, input_channels, output_channels, padding=padding, bias_init=bias_init, weight_init=weight_init)
    # Generate test data
    test_model.gen_test_data(height, width)
    # Save model
    test_model.save_core_model()
    # Populate converters
    test_model.populate_converters()


def initializer_args_handler(args):
    def check_unif_init_params(param_unif):
        if param_unif:
            if len(param_unif) != 2:
                raise argparse.ArgumentTypeError(
                    'The unif_init argument must consist of 2 numbers indicating a range.')
            if param_unif[0] > param_unif[1]:
                raise argparse.ArgumentTypeError(
                    'The unif_init argument requires the first value to be lesser than the second.')
    def check_const_init_params(const_param):
        if len(const_param) > 1:
            raise argparse.ArgumentTypeError(
                'The const_init argument must consist of 1 float number or none, in wich case, ' +
                'the default value will be used.'
            )
    initializers_types = ['unif', 'const'] # TODO make it an enum
    initializers = {'weight_init': DEFAULT_UNIF_INIT if args.seed_init is None else tf.random_uniform_initializer(
        *_DEFAULT_UNIF_INIT, args.seed_init),
                    'bias_init': DEFAULT_CONST_INIT}
    for k in initializers:
        values = vars(args)[k] if k in vars(args) else []  # Initializer values in the dictionary of arguments else use default
        if values:  # there is something to do
            if values[0] in initializers_types:  # First value of the arguments must be valid
                params =  values[1:]
                if values[0].lower() == 'unif':  # handle uniform
                    initializers[k] = tf.random_uniform_initializer(
                        *(float(e) for e in params) if check_unif_init_params(params) or params else _DEFAULT_UNIF_INIT,
                        args.seed_init
                    )
                elif values[0].lower() == 'const': # handle constant
                    initializers[k] = tf.constant_initializer(
                        float(params) if check_const_init_params(params) or params else _DEFAULT_CONST_INIT
                    )
            else: # the first argument wasn't a string or None
                raise argparse.ArgumentTypeError(
                    f'A type of initializer must be selected for {k}: "unif", "const" or None in which case, '+
                    'the default value for the initializer will be used.'
                    # TODO: ENUM with the initializer types
                )
        else: # use default and jump to the next initializer
            continue
    logging.debug('\n' + '\n'.join(f'{k} configuration: {initializers[k].get_config()}' for k in initializers))
    return initializers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Number of input channels')
    parser.add_argument(
        '-out', '--outputs', type=int, default=DEFAULT_OUTPUTS,
        help='Number of output channels')
    parser.add_argument(
        '-hi', '--height', type=int, default=DEFAULT_HEIGHT,
        help='Height of input image')
    parser.add_argument(
        '-wi', '--width', type=int, default=DEFAULT_WIDTH,
        help='Width of input image')
    parser.add_argument(
        '-kh', '--kernel_height', type=int, default=DEFAULT_KERNEL_HEIGHT,
        help='Height of kernel')
    parser.add_argument(
        '-kw', '--kernel_width', type=int, default=DEFAULT_KERNEL_WIDTH,
        help='Width of kernel')
    parser.add_argument(
        '-pd', '--padding', type=str, default=DEFAULT_PADDING,
        help='Padding mode')
    parser.add_argument(
        '--bias_init', nargs='*', default=argparse.SUPPRESS,
        help='Help'
    )
    parser.add_argument(
        '--weight_init', nargs='*', default=argparse.SUPPRESS,
        help='Help'
    )
    parser.add_argument(
        '--bias_const_init', type=float, default=argparse.SUPPRESS,
        help=f'Initialize bias with a constant. (default: {_DEFAULT_CONST_INIT})')
    parser.add_argument(
        '--bias_unif_init', nargs='+', type=float, default=argparse.SUPPRESS,
        help='Initialize bias with a random uniform distribution delimited '
             'by the range given by min and max values. '
        'If not specified, bias_const_init will be used instead.')
    parser.add_argument(
        '--weight_const_init', type=float, default=argparse.SUPPRESS,
        help='Initialize weights with a constant. '
        'If not specified, weight_unif_init will be used instead.')
    parser.add_argument(
        '--weight_unif_init', nargs='+', type=float, default=argparse.SUPPRESS,
        help='Initialize weights with a random uniform distribution delimited '
             f'by the range given by min and max values. (default: {_DEFAULT_UNIF_INIT})')
    parser.add_argument(
        '--seed_init', type=int,
        help='Set the seed value for the initializers.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs, output_channels=args.outputs,
         K_h=args.kernel_height, K_w=args.kernel_width,
         height=args.height, width=args.width,
         padding=args.padding,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'])
