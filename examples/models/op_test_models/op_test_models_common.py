# Copyright (c) 2020, XMOS Ltd, All rights reserved

import argparse
import logging
from enum import Enum
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


class DefaultOpTestModel(KerasModel):
    """
    Common class for those model that don't need to be trained
    with the default option to generate input data according to an input initializer
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
        self.data["export_data"], self.data["quant"] = input_initializers(
            self.input_init, *self.input_shape)
        # logging.debug(f'EXPORT DATA SAMPLE:\n{self.data["export_data"][4][0]}')
        # logging.debug(f'QUANT DATA SAMPLE:\n{self.data["quant"][0][0]}')

    def run(self):
        # TODO: Consider including this in model_generation.interface
        self.gen_test_data()
        self.save_core_model()
        self.populate_converters()


class DefaultOpTestConvModel(DefaultOpTestModel):
    def build_core_model(
            self, K_h, K_w, height, width, input_channels, output_channels, *,
            padding, bias_init, weight_init, input_init):
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
                                           kernel_initializer=weight_init)
                ])
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e


class OpTestDeepoutConvModel(DefaultOpTestConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, _, output_channels = args
        assert output_channels % 16 == 0, "# of output channels must be multiple of 16"
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        super().build_core_model(*args, **kwargs)


class OpTestInitializers(Enum):
    UNIF = "unif"
    CONST = "const"


# TODO: generalize this to produce a tensor of any rank between 2 and 4
def input_initializers(init, *args, batch=100, subset_len=10):
    assert batch >= subset_len, "Example subset cannot be larger than the full quantization set"
    height, width, channels = args[:3]  # pylint: disable=unbalanced-tuple-unpacking
    data = init(shape=(batch, height, width, channels), dtype="float32").numpy()
    subset = data[:subset_len, :, :, :]
    return subset, data


# TODO: this should be part of parse_args in the appropriate class
def strides_pool_arg_handler(args):
    parameters = {
        "strides": (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH),
        "pool_size": (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH),
    }
    arguments = {k: vars(args)[k] for k in parameters if k in vars(args)}
    for k in arguments:
        params = arguments[k]
        if len(params) > 2:
            raise argparse.ArgumentTypeError(
                f"The {k} argument must be at most 2 numbers.")
        else:
            arguments[k] = tuple(params) if len(params) == 2 else (params[0],) * 2

    return arguments


# TODO: this should be part of parse_args in the appropriate class
def initializer_args_handler(args):
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

    utils.set_all_seeds(args.seed)  # NOTE: All seeds initialized here
    initializers = {
        "alpha_init": tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed),
        "weight_init": tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed),
        "bias_init": DEFAULT_CONST_INIT,
        "input_init": tf.random_uniform_initializer(*_DEFAULT_UNIF_INIT, args.seed),
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
                *(params if params else _DEFAULT_UNIF_INIT), args.seed)
        elif init_type is OpTestInitializers.CONST:
            check_const_init_params(params)
            initializers[k] = tf.constant_initializer(
                params[0] if params else _DEFAULT_CONST_INIT)

    for k in initializers:
        logging.debug(
            f"{k} configuration: "
            f"{type(initializers[k]).__name__} {initializers[k].get_config()}"
        )

    return initializers


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


class OpTestParserInputInitializerMixin():
    def add_initializers(self):
        self.add_argument(
            "--seed", type=int,
            help="Set the seed value for the initializers."
        )
        self.add_argument(
            "--input_init", nargs="*", default=argparse.SUPPRESS,
            help="Initialize inputs. Possible initializers are: "
                 "const [CONST_VAL] or unif [MIN MAX]. "
                 f"(default: {OpTestInitializers.UNIF.value} "
                 f"{_DEFAULT_UNIF_INIT[0]} {_DEFAULT_UNIF_INIT[1]})",
        )


class OpTestParserInitializerMixin(OpTestParserInputInitializerMixin):
    def add_initializers(self):
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


#  for models with 2D dimensionality and padding
class OpTestDimParser(OpTestDefaultParser, OpTestParserInputInitializerMixin):
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
        else:
            self.add_initializers()


class OpTestPoolParser(OpTestDimParser, OpTestParserInputInitializerMixin):
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
        self.add_initializers()


class OpTestConvParser(OpTestParserInitializerMixin, OpTestDimParser):
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
        self.add_initializers()


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
        self.add_initializers()


# TODO: this should be part of DefaultOpTestFCModel as run
def run_main_fc(model, *, train_new_model, input_dim, output_dim, bias_init,
                weight_init, batch_size, epochs):
    if train_new_model:
        model.build(input_dim, output_dim,
                    bias_init=bias_init,
                    weight_init=weight_init)
        model.prep_data()
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
    model.gen_test_data()
    model.populate_converters()


# TODO: this should be part of DefaultOpTestConvModel as run
def run_main_conv(model, *, num_threads, input_channels, output_channels,
                  height, width, K_h, K_w, padding, bias_init, weight_init,
                  input_init):
    model.build(K_h, K_w, height, width, input_channels, output_channels,
                padding=padding,
                bias_init=bias_init,
                weight_init=weight_init,
                input_init=input_init)

    # TODO: this debug logging should probably live in model.build
    # for layer in model.core_model.layers:
    #     logging.debug(f"WEIGHT DATA SAMPLE:\n{layer.get_weights()[0][1]}")
    #     logging.debug(f"BIAS DATA SAMPLE:\n{layer.get_weights()[1]}")

    model.gen_test_data()
    model.save_core_model()
    model.populate_converters(
        xcore_num_threads=num_threads if num_threads else None)
