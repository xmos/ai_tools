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



class OpTestDefaultParser(argparse.ArgumentParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-path', nargs='?', default=defaults['path'],
            help='Path to a directory where models and data will be saved in subdirectories.')
        self.add_argument(
            '-v', '--verbose', action='store_true', default=False,
            help='Verbose mode.')
    
    def add_initializers(self):
        self.add_argument(
            '--bias_init', nargs='*', default=argparse.SUPPRESS,
            help='Initialize bias. Possible initializers are: const init or None.'
                f'(default: {OpTestInitializers.CONST.value} {_DEFAULT_CONST_INIT})'
        )
        self.add_argument(
            '--weight_init', nargs='*', default=argparse.SUPPRESS,
            help='Initialize weights. Possible initializers are: const, unif or None.'
                f'(default: {OpTestInitializers.UNIF.value} {_DEFAULT_UNIF_INIT})'
        )
        self.add_argument(
            '--input_init', nargs='*', default=argparse.SUPPRESS,
            help='Initialize inputs. Possible initializers are: const, unif or None.'
                f'(default: {OpTestInitializers.UNIF.value} {_DEFAULT_UNIF_INIT})'
        )
        self.add_argument(
            '--seed', type=int,
            help='Set the seed value for the initializers.'
        )

#  for models with 2D dimensionality and padding
class OpTestDimParser(OpTestDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '-in', '--inputs', type=int, default=defaults['inputs'],
            help='Number of input channels')
        self.add_argument(
            '-hi', '--height', type=int, default=defaults['height'],
            help='Height of input image')
        self.add_argument(
            '-wi', '--width', type=int, default=defaults['width'],
            help='Width of input image')
        self.add_argument(
            '-pd', '--padding', type=str, default=defaults['padding'],
            help='Padding mode')

#  for conv models 
class OpTestConvParser(OpTestDimParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '-out', '--outputs', type=int, default=defaults['outputs'],
            help='Number of output channels')
        self.add_argument(
            '-kh', '--kernel_height', type=int, default=defaults['kernel_height'],
            help='Height of kernel')
        self.add_argument(
            '-kw', '--kernel_width', type=int, default=defaults['kernel_width'],
            help='Width of kernel')
        self.add_initializers()

#  for fc models
class OpTestFcParser(OpTestDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--use_gpu', action='store_true', default=False,
            help='Use GPU for training. Might result in non-reproducible results.')
        self.add_argument(
            '-out', '--output_dim', type=int, default=defaults['output_dim'],
            help='Number of output dimensions, must be at least 2.')
        self.add_argument(
            '-in', '--input_dim', type=int, default=defaults['input_dim'],
            help='Input dimension, must be multiple of 32.')
        self.add_argument(
            '--train_model', action='store_true', default=False,
            help='Train new model instead of loading pretrained tf.keras model.')
        self.add_initializers()


def run_main_fc(model, *, train_new_model, input_dim, output_dim, bias_init, weight_init, batch_size, epochs):
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
                     padding=padding, bias_init=bias_init, weight_init=weight_init, input_init=input_init)
    
    # to check if the inits work
    for layer in model.core_model.layers: logging.debug(f'WEIGHT DATA SAMPLE:\n{layer.get_weights()[0][1]}') # weights
    for layer in model.core_model.layers: logging.debug(f'BIAS DATA SAMPLE:\n{layer.get_weights()[1]}') # bias
    
    # Generate test data
    model.gen_test_data(height, width)
    logging.debug(f'EXPORT DATA SAMPLE:\n{model.data["export_data"][4][0]}')
    logging.debug(f'QUANT DATA SAMPLE:\n{model.data["quant"][0][0]}')
    # Save model
    model.save_core_model()
    # Populate converters
    model.populate_converters(xcore_num_threads=num_threads if num_threads else None)
