# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import os
import re
import pathlib
import random
import argparse
import sys
import importlib
import numpy as np
from typing import NamedTuple, Union

from tflite2xcore import xlogging as logging


class QuantizationTuple(NamedTuple):
    scale: float
    zero_point: int


def lazy_import(fullname):
    try:
        return sys.modules[fullname]
    except KeyError:
        pass
    # parent module is loaded eagerly
    spec = importlib.util.find_spec(fullname)
    try:
        lazy_loader = importlib.util.LazyLoader(spec.loader)
    except AttributeError:
        return lazy_import(fullname)

    module = importlib.util.module_from_spec(spec)
    lazy_loader.exec_module(module)
    sys.modules[fullname] = module
    if "." in fullname:
        parent_name, _, child_name = fullname.rpartition(".")
        parent_module = sys.modules[parent_name]
        setattr(parent_module, child_name, module)
    return module


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf = lazy_import("tensorflow")

VE, ACC_PERIOD, WORD_SIZE = 32, 16, 4
DEFAULT_SEED = 123


def set_all_seeds(seed=DEFAULT_SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_gpu_usage(use_gpu, verbose):
    # can throw annoying error if CUDA cannot be initialized
    default_log_level = os.environ["TF_CPP_MIN_LOG_LEVEL"]
    if not verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = default_log_level

    if gpus:
        if use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable=True)
        else:
            logging.getLogger().info("GPUs disabled.")
            tf.config.experimental.set_visible_devices([], "GPU")
    elif use_gpu:
        logging.getLogger().warning("No available GPUs found, defaulting to CPU.")
    logging.getLogger().debug(f"Eager execution enabled: {tf.executing_eagerly()}")


class VerbosityParser(argparse.ArgumentParser):
    def __init__(self, *args, verbosity_config=None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        kwargs.setdefault("conflict_handler", "resolve")
        super().__init__(*args, **kwargs)

        verbosity_config = verbosity_config or dict()
        verbosity_config.setdefault("action", "count")
        verbosity_config.setdefault("default", 0)
        verbosity_config.setdefault(
            "help",
            "Set verbosity level. "
            "-v: info on passes matching; -vv: general debug info; "
            "-vvv: extra debug info including some intermediate mutation results.",
        )
        self.add_argument("-v", "--verbose", **verbosity_config)

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        logging.set_verbosity(args.verbose)
        set_gpu_usage(args.use_gpu if hasattr(args, "use_gpu") else False, args.verbose)
        return args


# TODO: remove this
def convert_path(path):
    if isinstance(path, pathlib.Path):
        return path
    elif isinstance(path, str):
        return pathlib.Path(path)
    else:
        raise TypeError(f"Expected path of type str or pathlib.Path, got {type(path)}")


def snake_to_camel(word: str) -> str:
    output = "".join(x.capitalize() or "_" for x in word.split("_"))
    return output[0].lower() + output[1:]


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def quantize(
    arr: "np.ndarray",
    scale: float,
    zero_point: int,
    dtype: Union[type, "np.dtype"] = np.int8,
) -> "np.ndarray":
    t = np.round(np.float32(arr) / np.float32(scale)).astype(np.int32) + zero_point
    return dtype(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max))


def dequantize(arr: "np.ndarray", scale: float, zero_point: int) -> "np.ndarray":
    return np.float32(arr.astype(np.int32) - np.int32(zero_point)) * np.float32(scale)
