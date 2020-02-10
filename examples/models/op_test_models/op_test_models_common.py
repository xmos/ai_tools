# Copyright (c) 2020, XMOS Ltd, All rights reserved

import argparse
import logging
from enum import Enum
import tensorflow as tf
from tflite2xcore.model_generation import utils
import numpy as np

_DEFAULT_CONST_INIT = 0
_DEFAULT_UNIF_INIT = [-1, 1]
DEFAULT_CONST_INIT = tf.constant_initializer(_DEFAULT_CONST_INIT)
DEFAULT_UNIF_INIT = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT)


class OpTestInitializers(Enum):
    UNIF = 'unif'
    CONST = 'const'


def input_initializers(init, height, width, channels, *, batch=100):
    # NOTE: same but with initializes as in utils.generate_dummy_data
    data = init(shape=(batch, height, width, channels), dtype='float32').numpy()
    subset = np.concatenate(
        [np.zeros((1, height, width, channels), dtype=np.float32),
         np.ones((1, height, width, channels), dtype=np.float32),
         data[:8, :, :, :]],  # pylint: disable=unsubscriptable-object
        axis=0)
    return subset, data


def initializer_args_handler(args):
    def check_unif_init_params(param_unif):
        if param_unif:
            if len(param_unif) != 2:
                raise argparse.ArgumentTypeError(
                    "The 'unif' initialization argument requires 2 numbers indicating a range.")
            if param_unif[0] > param_unif[1]:
                raise argparse.ArgumentTypeError(
                    "The 'unif' initialization argument requires the first value to be lesser than the second.")

    def check_const_init_params(const_param):
        if len(const_param) > 1:
            raise argparse.ArgumentTypeError(
                'The const argument must consist of 1 float number or '
                'none, in wich case, the default value will be used.'
            )
    utils.set_all_seeds(args.seed) #NOTE All seeds initialized here
    initializers = {
        'weight_init': tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed),
        'bias_init': DEFAULT_CONST_INIT,
        'input_init': tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed)
    }

    init_args = {k: vars(args)[k] for k in initializers if k in vars(args)}
    for k, arg_params in init_args.items():
        try:
            init_type = OpTestInitializers(arg_params[0].lower())
        except (IndexError, ValueError):
            raise argparse.ArgumentTypeError(
                f"A type of initializer for {k} must be selected from "
                f"{[v.value for v in OpTestInitializers]}."
            )

        try:
            params = [float(n) for n in arg_params[1:]]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid numeric parameter(s) {arg_params[1:]} for "
                f"{k} {init_type.value} initialization"
            )

        if init_type is OpTestInitializers.UNIF:
            check_unif_init_params(params)
            initializers[k] = tf.random_uniform_initializer(
                *(params if params else _DEFAULT_UNIF_INIT), args.seed
            )
        elif init_type is OpTestInitializers.CONST:
            check_const_init_params(params)
            initializers[k] = tf.constant_initializer(
                params[0] if params else _DEFAULT_CONST_INIT
            )

    for k in initializers:
        logging.debug(
            f"{k} configuration: {type(initializers[k]).__name__} {initializers[k].get_config()}")

    return initializers


def parser_add_initializers(parser):
    parser.add_argument(
        '--bias_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize bias. Possible initializers are: const init or None.'
             f'(default: {OpTestInitializers.CONST.value} {_DEFAULT_CONST_INIT})'
    )
    parser.add_argument(
        '--weight_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize weights. Possible initializers are: const, unif or None.'
             f'(default: {OpTestInitializers.UNIF.value} {_DEFAULT_UNIF_INIT})'
    )
    parser.add_argument(
        '--input_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize inputs. Possible initializers are: const, unif or None.'
             f'(default: {OpTestInitializers.UNIF.value} {_DEFAULT_UNIF_INIT})'
    )
    parser.add_argument(
        '--seed', type=int,
        help='Set the seed value for the initializers.'
    )
    return parser


def get_default_parser(is_fc=False, **kwargs,):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-path', nargs='?', default=kwargs['DEFAULT_PATH'],
        help='Path to a directory where models and data will be saved in subdirectories.')
    if not is_fc:
        parser.add_argument(
            '-in', '--inputs', type=int, default=kwargs['DEFAULT_INPUTS'],
            help='Number of input channels')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    return parser


def get_dim_parser(**kwargs):  # for models with 2D dimensionality and padding
    parser = get_default_parser(**kwargs)
    parser.add_argument(
        '-hi', '--height', type=int, default=kwargs['DEFAULT_HEIGHT'],
        help='Height of input image')
    parser.add_argument(
        '-wi', '--width', type=int, default=kwargs['DEFAULT_WIDTH'],
        help='Width of input image')
    parser.add_argument(
        '-pd', '--padding', type=str, default=kwargs['DEFAULT_PADDING'],
        help='Padding mode')
    return parser


def get_conv_parser(**kwargs): # for the conv models
    parser = get_dim_parser(**kwargs)
    parser.add_argument(
        '-out', '--outputs', type=int, default=kwargs['DEFAULT_OUTPUTS'],
        help='Number of output channels')
    parser.add_argument(
        '-kh', '--kernel_height', type=int, default=kwargs['DEFAULT_KERNEL_HEIGHT'],
        help='Height of kernel')
    parser.add_argument(
        '-kw', '--kernel_width', type=int, default=kwargs['DEFAULT_KERNEL_WIDTH'],
        help='Width of kernel')
    parser = parser_add_initializers(parser)
    return parser


def get_fc_parser(**kwargs): # for the fc models
    parser = get_default_parser(is_fc=True, **kwargs)
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results.')
    parser.add_argument(
        '-out', '--output_dim', type=int, default=kwargs['DEFAULT_OUTPUT_DIM'],
        help='Number of output dimensions, must be at least 2.')
    parser.add_argument(
        '-in', '--input_dim', type=int, default=kwargs['DEFAULT_INPUT_DIM'],
        help='Input dimension, must be multiple of 32.')
    parser.add_argument(
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
    parser = parser_add_initializers(parser)
    return parser

# For FC models
def run_main(model, *, train_new_model, input_dim, output_dim, bias_init, weight_init, batch_size, epochs):
    if train_new_model:
        # Build model and compile
        model.build(input_dim, output_dim,
                         bias_init=bias_init, weight_init=weight_init)
        # Prepare training data
        model.prep_data()
        # Train model
        model.train(batch_size=batch_size, epochs=epochs)
        model.save_core_model()
    else:
        # Recover previous state from file system
        model.load_core_model()
        if output_dim and output_dim != model.output_dim:
            raise ValueError(
                f"specified output_dim ({output_dim}) "
                f"does not match loaded model's output_dim ({model.output_dim})"
            )
    # Generate test data
    model.gen_test_data()
    # Populate converters
    model.populate_converters()

# For conv models
def run_main_conv(model, *, num_threads, input_channels, output_channels,
                  height, width, K_h, K_w, padding, bias_init, weight_init, input_init):
    # Instantiate model
    # Build model and compile
    model.build(K_h, K_w, height, width, input_channels, output_channels,
                     padding=padding, bias_init=bias_init, weight_init=weight_init)
    
    # to check if the inits work
    for layer in model.core_model.layers: logging.debug(f'WEIGHT DATA SAMPLE:\n{layer.get_weights()[0][1]}') # weights
    for layer in model.core_model.layers: logging.debug(f'BIAS DATA SAMPLE:\n{layer.get_weights()[1]}') # bias
    
    # Generate test data
    model.gen_test_data(height, width, input_init)
    logging.debug(f'EXPORT DATA SAMPLE:\n{model.data["export_data"][4][0]}')
    logging.debug(f'QUANT DATA SAMPLE:\n{model.data["quant"][0][0]}')
    # Save model
    model.save_core_model()
    # Populate converters
    model.populate_converters(xcore_num_threads=num_threads if num_threads else None)
