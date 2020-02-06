import argparse
import logging
import tensorflow as tf

_DEFAULT_CONST_INIT = 0
_DEFAULT_UNIF_INIT = [-1, 1]
DEFAULT_CONST_INIT = tf.constant_initializer(_DEFAULT_CONST_INIT)
DEFAULT_UNIF_INIT = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT)

def initializer_args_handler(args):
    def check_unif_init_params(param_unif):
        if param_unif:
            if len(param_unif) != 2:
                raise argparse.ArgumentTypeError(
                    'The unif_init argument must consist of 2 numbers indicating a range.')
            if float(param_unif[0]) > float(param_unif[1]):
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
                        float(*params) if check_const_init_params(params) or params else _DEFAULT_CONST_INIT
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

def parser_add_initializers(parser):
    parser.add_argument(
        '--bias_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize bias. Possible initializers are: const init or None.'
        f'(default: const {_DEFAULT_CONST_INIT})'  # TODO: ENUM for initializer types
    )
    parser.add_argument(
        '--weight_init', nargs='*', default=argparse.SUPPRESS,
        help='Initialize weights. Possible initializers are: const, unif or None.'
        f'(default: uniform {_DEFAULT_UNIF_INIT})'  # TODO ENUM for initializer types
    )
    parser.add_argument(
        '--seed_init', type=int,
        help='Set the seed value for the initializers.'
    )
    return parser
