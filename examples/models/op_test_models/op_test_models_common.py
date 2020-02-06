# Copyright (c) 2020, XMOS Ltd, All rights reserved

import argparse
import logging
from enum import Enum
import tensorflow as tf

_DEFAULT_CONST_INIT = 0
_DEFAULT_UNIF_INIT = [-1, 1]
DEFAULT_CONST_INIT = tf.constant_initializer(_DEFAULT_CONST_INIT)
DEFAULT_UNIF_INIT = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT)


class OpTestInitializers(Enum):
    UNIF = 'unif'
    CONST = 'const'


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

    initializers = {
        'weight_init': tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed),
        'bias_init': DEFAULT_CONST_INIT
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
        f'(default: {str(OpTestInitializers.CONST.value)} {_DEFAULT_CONST_INIT})'
    )
    parser.add_argument(
        '--weight_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize weights. Possible initializers are: const, unif or None.'
        f'(default: {OpTestInitializers.UNIF.value} {_DEFAULT_UNIF_INIT})'
    )
    parser.add_argument(
        '--seed', type=int,
        help='Set the seed value for the initializers.'  # TODO: generalize to set all seeds with this
    )
    return parser
