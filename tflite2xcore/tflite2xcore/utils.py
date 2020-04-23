# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import os
import pathlib
import random
import argparse
import sys
import importlib

import numpy as np

from tflite2xcore import xlogging as logging


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


def convert_path(path):
    if isinstance(path, pathlib.Path):
        return path
    elif isinstance(path, str):
        return pathlib.Path(path)
    else:
        raise TypeError(f"Expected path of type str or pathlib.Path, got {type(path)}")
