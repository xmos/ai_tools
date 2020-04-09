# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

# best to import this before tf
from model_common import DefaultParser, InitializerParser, TrainableParser

import argparse
import numpy as np
import tensorflow as tf
from enum import Enum
from abc import abstractmethod
from tflite2xcore.model_generation.interface import KerasModel


_DEFAULT_CONST_INIT = 0
_DEFAULT_UNIF_INIT = [-1, 1]
DEFAULT_CONST_INIT = tf.constant_initializer(_DEFAULT_CONST_INIT)
DEFAULT_UNIF_INIT = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT)
DEFAULT_STRIDE_HEIGHT = 1
DEFAULT_STRIDE_WIDTH = 2
DEFAULT_POOL_HEIGHT = 3
DEFAULT_POOL_WIDTH = 5


#  ----------------------------------------------------------------------------
#                                  MODELS
#  ----------------------------------------------------------------------------

class OpTestDefaultModel(KerasModel):
    """
    Common class for those models that don't need to be trained and use an
    input initializer to generate input data
    """
    @abstractmethod
    def build_core_model(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        self._prep_backend()
        try:
            self.build_core_model(*args, **kwargs)
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise e from None
        self.core_model.build()
        self.core_model.summary()

    def train(self):
        pass

    def prep_data(self):
        pass

    def gen_test_data(self, batch=100, subset_len=10):
        assert self.input_shape, "To generate test data this model needs an input shape"
        assert self.input_init, "To generate test data this model needs an input initializer"
        (())
        self.data['export'] = self.input_init(shape=(batch, *self.input_shape),
                                              dtype="float32").numpy()
        self.data['quant'] = self.data['export'][:subset_len]
        # TODO: use array log message helper from utils
        # self.logger.debug(f"data['export'] sample:\n{self.data['export'][-1]}")
        # self.logger.debug(f"data['quant'] sample:\n{self.data['quant'][-1]}")

    def run(self, *, num_threads=None):
        self.save_core_model()
        self.convert_and_save(xcore_num_threads=num_threads)


class OpTestDefaultConvModel(OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, input_channels, output_channels, *,
            padding, strides=(1,1), **inits):
        self.input_init = inits['input_init']
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Conv2D(filters=output_channels,
                                       kernel_size=(K_h, K_w),
                                       padding=padding,
                                       strides=strides,
                                       input_shape=(height, width, input_channels),
                                       bias_initializer=inits['bias_init'],
                                       kernel_initializer=inits['weight_init'])
            ]
        )

class OpTestPaddedConvModel(OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, input_channels, output_channels, *,
            padding, strides=(1,1), **inits):
        self.input_init = inits['input_init']
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.ZeroPadding2D(padding=padding,
                                              input_shape=(height, width, output_channels)),
                tf.keras.layers.Conv2D(filters=output_channels,
                                       kernel_size=(K_h, K_w),
                                       padding='valid',
                                       strides=strides,
                                       input_shape=(height, width, input_channels),
                                       bias_initializer=inits['bias_init'],
                                       kernel_initializer=inits['weight_init'])
            ]
        )


class OpTestDefaultFCModel(KerasModel):
    def build(self, input_dim, output_dim, **inits):
        super().build()

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(input_dim, 1, 1)),
                tf.keras.layers.Dense(output_dim, activation='softmax',
                                      bias_initializer=inits['bias_init'],
                                      kernel_initializer=inits['weight_init'])
            ]
        )
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        self.core_model.summary()

    @property
    def input_dim(self):
        return self.input_shape[0]

    @property
    def output_dim(self):
        return self.output_shape[0]

    # For training
    def prep_data(self):
        self.data = generate_fake_lin_sep_dataset(
            self.output_dim, self.input_dim,
            train_samples_per_class=51200//self.output_dim,
            test_samples_per_class=10240//self.output_dim)

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        subset_inds = np.searchsorted(
            self.data['y_test'].flatten(), np.arange(self.output_dim))
        self.data['export'] = self.data['x_test'][subset_inds]  # pylint: disable=unsubscriptable-object
        self.data['quant'] = self.data['x_train']

    def convert_to_stripped(self, **converter_args):
        converter_args.setdefault('remove_softmax', True)
        super().convert_to_stripped(**converter_args)

    def convert_to_xcore(self, **converter_args):
        converter_args.setdefault('remove_softmax', True)
        super().convert_to_xcore(**converter_args)

    def build_and_train(self, input_dim, output_dim, batch_size, epochs, **inits):
        self.build(input_dim, output_dim, **inits)
        self.prep_data()
        self.train(batch_size=batch_size, epochs=epochs)
        self.save_core_model()

    def load_core_model(self, output_dim):
        super().load_core_model()
        if output_dim and output_dim != self.output_dim:
            raise ValueError(
                f"specified output_dim ({output_dim}) "
                f"does not match loaded model's output_dim ({self.output_dim})"
            )


# TODO: move this to model_generation utils
def generate_fake_lin_sep_dataset(classes=2, dim=32, *,
                                  train_samples_per_class=5120,
                                  test_samples_per_class=1024):
    z = np.linspace(0, 2*np.pi, dim)

    # generate data and class labels
    x_train, x_test, y_train, y_test = [], [], [], []
    for j in range(classes):
        mean = np.sin(z) + 10*j/classes
        cov = 10 * np.diag(.5*np.cos(j * z) + 2) / (classes-1)
        x_train.append(
            np.random.multivariate_normal(
                mean, cov, size=train_samples_per_class))
        x_test.append(
            np.random.multivariate_normal(
                mean, cov, size=test_samples_per_class))
        y_train.append(j * np.ones((train_samples_per_class, 1)))
        y_test.append(j * np.ones((test_samples_per_class, 1)))

    # stack arrays
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    # normalize
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # expand dimensions for TFLite compatibility
    def expand_array(arr):
        return np.reshape(arr, arr.shape + (1, 1))
    x_train = expand_array(x_train)
    x_test = expand_array(x_test)

    return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
            'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}


#  ----------------------------------------------------------------------------
#                                   PARSERS
#  ----------------------------------------------------------------------------

class OpTestInitializers(Enum):
    UNIF = "unif"
    CONST = "const"


class OpTestInitializerParser(InitializerParser):
    __INIT_HELPERS = {
        'input_init': "Initializer for input data distribution.",
        'weight_init': "Initializer for weight distribution.",
        'bias_init': "Initializer for bias distribution."
    }

    def _default_handler(self, defaults):
        self.default_inits = defaults["inits"]
        for init_name, init_settings in self.default_inits.items():
            init_type = init_settings['type']
            assert init_type in OpTestInitializers

            def_str = f"{init_type} "
            if init_type is OpTestInitializers.UNIF:
                def_str += f"{_DEFAULT_UNIF_INIT[0]} {_DEFAULT_UNIF_INIT[1]}"
            elif init_type is OpTestInitializers.CONST:
                def_str += f"{_DEFAULT_CONST_INIT}"

            try:
                init_help = init_settings['help']
            except KeyError:
                try:
                    init_help = self.__INIT_HELPERS[init_name]
                except KeyError:
                    init_help = "[MISSING INITIALIZER DESCRIPTION]"

            self.add_argument(
                f"--{init_name}", nargs="*", default=argparse.SUPPRESS,
                help=f"{init_help} "
                     "Possible initializers are: const [CONST_VAL] or unif [MIN MAX]. "
                     f"(default: {def_str})"
            )

    def _initializer_args_handler(self, args):
        def check_unif_init_params(param_unif):
            if param_unif:
                if len(param_unif) != 2:
                    raise argparse.ArgumentTypeError(
                        "The 'unif' initialization argument requires "
                        "2 numbers indicating a range."
                    )
                if param_unif[0] > param_unif[1]:
                    raise argparse.ArgumentTypeError(
                        "The 'unif' initialization argument requires "
                        "the first value to be lesser than the second."
                    )

        def check_const_init_params(const_param):
            if len(const_param) > 1:
                raise argparse.ArgumentTypeError(
                    "The const argument must consist of 1 float number or "
                    "none, in wich case, the default value will be used."
                )

        def instantiate_default_init(init_type):
            if init_type is OpTestInitializers.UNIF:
                init = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, self.seed)
            else:
                init = DEFAULT_CONST_INIT
            return init

        initializers = {k: instantiate_default_init(init_settings['type'])
                        for k, init_settings in self.default_inits.items()}
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
                    *(params if params else _DEFAULT_UNIF_INIT), self.seed)
            elif init_type is OpTestInitializers.CONST:
                check_const_init_params(params)
                initializers[k] = tf.constant_initializer(
                    params[0] if params else _DEFAULT_CONST_INIT)

        for k in initializers:
            self.logger.debug(
                f"{k} configuration: "
                f"{type(initializers[k]).__name__} {initializers[k].get_config()}"
            )
        # initializers = {k: v for k, v in initializers.items() if k in vars(args)}
        args.inits = initializers


class OpTestImgParser(OpTestInitializerParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-in", "--inputs", type=int, default=defaults["inputs"],
            help="Number of input channels",
        )
        self.add_argument(
            "-hi", "--height", type=int, default=defaults["height"],
            help="Height of input image",
        )
        self.add_argument(
            "-wi", "--width", type=int, default=defaults["width"],
            help="Width of input image",
        )
        if 'padding' in defaults:
            self.add_argument(
                "-pd", "--padding", type=str, default=defaults["padding"],
                choices=['same', 'valid'],
                help="Padding mode",
            )


# TODO: after the conv2d enhancements, this should be used in the conv parsers
class OpTestPoolParser(OpTestImgParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-st", "--strides", nargs="+", type=int, default=argparse.SUPPRESS,
            help=f"Strides, vertical first (default: {defaults['strides']})",
        )
        self.add_argument(
            "-po", "--pool_size", nargs="+", type=int, default=argparse.SUPPRESS,
            help=f"Pool size:, vertical first (default: {defaults['pool_size']})",
        )

    # TODO: decouple handling of strides and pool size
    # TODO: this should be used for convolutions as well, and pool renamed to filter for generality
    def strides_pool_arg_handler(self, args):
        parameters = {
            "strides": (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH),
            "pool_size": (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH),
        }
        arguments = {k: vars(args)[k] if k in vars(args) else parameters[k]
                     for k in parameters}
        for k in arguments:
            params = arguments[k]
            if len(params) > 2:
                raise argparse.ArgumentTypeError(
                    f"The {k} argument must be at most 2 numbers.")
            else:
                arguments[k] = tuple(params) if len(params) == 2 else (params[0],) * 2

        return arguments

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        strides_pool = self.strides_pool_arg_handler(args)
        args.strides = strides_pool['strides']
        args.pool_size = strides_pool['pool_size']
        return args


class OpTestConvParser(OpTestImgParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-out", "--outputs", type=int, default=defaults["outputs"],
            help="Number of output channels",
        )
        self.add_argument(
            "-kh", "--kernel_height", type=int, default=defaults["kernel_height"],
            help="Height of kernel",
        )
        self.add_argument(
            "-kw", "--kernel_width", type=int, default=defaults["kernel_width"],
            help="Width of kernel",
        )


class OpTestFCParser(TrainableParser, OpTestInitializerParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-out", "--output_dim", type=int, default=defaults["output_dim"],
            help="Number of output dimensions, must be at least 2.",
        )
        self.add_argument(
            "-in", "--input_dim", type=int, default=defaults["input_dim"],
            help="Input dimension, must be multiple of 32.",
        )
