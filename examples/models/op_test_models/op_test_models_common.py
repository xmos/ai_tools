# Copyright (c) 2020, XMOS Ltd, All rights reserved

import argparse
import logging
from enum import Enum
from pathlib import Path
import tensorflow as tf
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import numpy as np
from abc import abstractmethod

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
        self.build_core_model(*args, **kwargs)
        self.core_model.build()
        self.core_model.summary()

    def train(self):
        pass

    def prep_data(self):
        pass

    def gen_test_data(self):
        assert self.input_shape, "To generate test data this model needs an input shape"
        assert self.input_init, "To generate test data this model needs an input initializer"
        self.data["export_data"], self.data["quant"] = input_initializer(
            self.input_init, *self.input_shape)
        # logging.debug(f'EXPORT DATA SAMPLE:\n{self.data["export_data"][4][0]}')
        # logging.debug(f'QUANT DATA SAMPLE:\n{self.data["quant"][0][0]}')

    def run(self):
        self.gen_test_data()
        self.save_core_model()
        self.populate_converters()


class OpTestDefaultConvModel(OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, input_channels, output_channels, *,
            padding, bias_init, weight_init, input_init):
        assert output_channels % 16 == 0, "# of output channels must be multiple of 16"
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        self.input_init = input_init
        try:
            self.core_model = tf.keras.Sequential(
                name=self.name,
                layers=[
                    tf.keras.layers.Conv2D(filters=output_channels,
                                           kernel_size=(K_h, K_w),
                                           padding=padding,
                                           input_shape=(height, width, input_channels),
                                           bias_initializer=bias_init,
                                           kernel_initializer=weight_init
                ])
            # for layer in self.core_model.layers:
            #     logging.debug(f"WEIGHT DATA SAMPLE:\n{layer.get_weights()[0][1]}")
            #     logging.debug(f"BIAS DATA SAMPLE:\n{layer.get_weights()[1]}")
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e

    def run(self, *, num_threads, input_channels, output_channels,
            height, width, K_h, K_w, padding, weight_init, bias_init, input_init):
        self.build(K_h, K_w, height, width, input_channels, output_channels, padding=padding,
                   weight_init=weight_init, bias_init=bias_init, input_init=intput_init)
        self.gen_test_data()
        self.save_core_model()
        self.populate_converters(
            xcore_num_threads=num_threads if num_threads else None)


class DefaultOpTestFCModel(KerasModel):
    def build(self, input_dim, output_dim,
              weight_init, bias_init):
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

    # For exports
    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        subset_inds = np.searchsorted(
            self.data['y_test'].flatten(), np.arange(self.output_dim))
        self.data['export_data'] = self.data['x_test'][subset_inds]
        self.data['quant'] = self.data['x_train']

    def to_tf_stripped(self):
        super().to_tf_stripped(remove_softmax=True)

    def to_tf_xcore(self):
        super().to_tf_xcore(remove_softmax=True)

    def run(self, *, train_new_model, input_dim, output_dim, weight_init, bias_init,
            batch_size, epochs):
        if train_new_model:
            self.build(input_dim, output_dim, weight_init, bias_init)
            self.prep_data()
            self.train(batch_size=batch_size, epochs=epochs)
            self.save_core_model()
        else:
            # Recover previous state from file system
            self.load_core_model()
            if output_dim and output_dim != self.output_dim:
                raise ValueError(
                    f"specified output_dim ({output_dim}) "
                    f"does not match loaded model's output_dim ({self.output_dim})"
                )
        self.gen_test_data()
        self.populate_converters()


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


# TODO: generalize this to produce a tensor of any rank between 2 and 4
def input_initializer(init, *args, batch=100, subset_len=10):
    assert batch >= subset_len, "Example subset cannot be larger than the full quantization set"
    height, width, channels = args[:3]  # pylint: disable=unbalanced-tuple-unpacking
    data = init(shape=(batch, height, width, channels), dtype="float32").numpy()
    subset = data[:subset_len, :, :, :]
    return subset, data


class OpTestDefaultParser(argparse.ArgumentParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-path", nargs="?", default=defaults["path"],
            help="Path to a directory where models and data will be saved in subdirectories.",
        )
        self.add_argument(
            "-v", "--verbose", action="store_true", default=False,
            help="Verbose mode."
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        utils.set_verbosity(args.verbose)
        args.path = Path(args.path)
        return args


class OpTestParserInputInitializerMixin(OpTestDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)

        self.seed = None
        self.add_argument(
            "--seed", type=int,
            help="Set the seed value for the initializers."
        )

        self.default_inits = defaults["inits"]
        for init_name, init_settings in self.default_inits.items():
            init_type = init_settings['type']
            assert init_type in OpTestInitializers

            def_str = f"{init_type} "
            if init_type is OpTestInitializers.UNIF:
                def_str += f"{_DEFAULT_UNIF_INIT[0]} {_DEFAULT_UNIF_INIT[1]}"
            elif init_type is OpTestInitializers.CONST:
                def_str += f"{_DEFAULT_CONST_INIT}"

            self.add_argument(
                f"--{init_name}", nargs="*", default=argparse.SUPPRESS,
                help=f"{init_settings['help']} "
                     "Possible initializers are: const [CONST_VAL] or unif [MIN MAX]. "
                     f"(default: {def_str})"
            )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self.seed = args.seed
        args.inits = self._initializer_args_handler(args)
        return args

    @property
    def _filtered_inits(self):
        """
        Returns a dictionary with names and default values (and initialized
        with a seed if specified) of the initializers declared on the default
        dict in the creation of the parser.
        """
        def instantiate_default_init(init_type):
            """
            Instantiates a initializer based on its type (OpTestInitializer).
            """
            if init_type is OpTestInitializers.UNIF:
                init = tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, self.seed)
            else:
                init = DEFAULT_CONST_INIT
            return init
        initializers = {}
        for k, init_settings in self.default_inits.items():
            initializers.update({k: instantiate_default_init(init_settings['type'])})
        return initializers

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

        initializers = self._filtered_inits
        # for k, init in initializers.items():
        #     print(f"{k}: {init}")
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
            logging.debug(
                f"{k} configuration: "
                f"{type(initializers[k]).__name__} {initializers[k].get_config()}"
            )
        # initializers = {k: v for k, v in initializers.items() if k in vars(args)}
        return initializers


class OpTestParserInitializerMixin(OpTestParserInputInitializerMixin):
    def add_initializers(self, seed=False):
        super().add_initializers()
        self.add_argument(
            "--bias_init", nargs="*", default=argparse.SUPPRESS,
            help="Initialize bias. Possible initializers are: "
                 "const [CONST_VAL] or unif [MIN MAX]. "
                 f"(default: {OpTestInitializers.CONST.value} {_DEFAULT_CONST_INIT})",
        )
        self.add_argument(
            "--weight_init", nargs="*", default=argparse.SUPPRESS,
            help="Initialize weights. Possible initializers are: "
                 "const [CONST_VAL] or unif [MIN MAX]. "
                 f"(default: {OpTestInitializers.UNIF.value} "
                 f"{_DEFAULT_UNIF_INIT[0]} {_DEFAULT_UNIF_INIT[1]})",
        )


class OpTestImgParser(OpTestParserInputInitializerMixin):
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
                help="Padding mode",
            )


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


class OpTestConvParser(OpTestParserInitializerMixin, OpTestImgParser):
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


class OpTestTrainableParser(OpTestDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "--train_model", action="store_true", default=False,
            help="Train new model instead of loading pretrained tf.keras model.",
        )
        self.add_argument(
            "--use_gpu", action="store_true", default=False,
            help="Use GPU for training. Might result in non-reproducible results.",
        )
        self.add_argument(
            "-bs", "--batch_size", type=int, default=defaults["batch_size"],
            help="Set the training batch size."
        )
        self.add_argument(
            "-ep", "--epochs", type=int, default=defaults["epochs"],
            help="Set the number of training epochs size."
        )


class OpTestFCParser(OpTestTrainableParser, OpTestParserInitializerMixin):
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
