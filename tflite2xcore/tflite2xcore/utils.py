# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import os
import re
import random
import argparse
import sys
import importlib
import logging
import numpy as np  # type: ignore
from functools import wraps
from types import ModuleType, TracebackType
from typing import (
    Union,
    Optional,
    Dict,
    Any,
    TypeVar,
    Callable,
    cast,
    Tuple,
    Type,
    NamedTuple,
)


class QuantizationTuple(NamedTuple):
    scale: float
    zero_point: int


# TODO: consider removing this after new integration tests are in
def lazy_import(fullname: str) -> ModuleType:
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

# -----------------------------------------------------------------------------
#                          XCORE MAGIC NUMBERS
# -----------------------------------------------------------------------------

VE, ACC_PERIOD, WORD_SIZE = 32, 16, 4


# -----------------------------------------------------------------------------
#                            REPRODUCIBILITY
# -----------------------------------------------------------------------------

DEFAULT_SEED = 123


def set_all_seeds(seed: int = DEFAULT_SEED) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_gpu_usage(use_gpu: bool, verbose: Union[bool, int]) -> None:
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
            logging.info("GPUs disabled.")
            tf.config.experimental.set_visible_devices([], "GPU")
    elif use_gpu:
        logging.warning("No available GPUs found, defaulting to CPU.")
    logging.debug(f"Eager execution enabled: {tf.executing_eagerly()}")


# -----------------------------------------------------------------------------
#                       LOGGING & STRING FORMATTING
# -----------------------------------------------------------------------------


def set_verbosity(verbosity: int = 0) -> None:
    verbosities = [logging.WARNING, logging.INFO, logging.DEBUG]
    verbosity = min(verbosity, len(verbosities) - 1)

    logging.basicConfig(level=verbosities[verbosity])
    if not verbosity:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)


class VerbosityParser(argparse.ArgumentParser):
    def __init__(
        self,
        *args: Any,
        verbosity_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
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
            "-v: summary info of mutations; -vv: detailed mutation and debug info.",
        )
        self.add_argument("-v", "--verbose", **verbosity_config)

    def parse_args(self, *args, **kwargs):  # type: ignore
        args = super().parse_args(*args, **kwargs)
        set_verbosity(args.verbose)  # type: ignore
        set_gpu_usage(args.use_gpu if hasattr(args, "use_gpu") else False, args.verbose)  # type: ignore
        return args


def snake_to_camel(word: str) -> str:
    output = "".join(x.capitalize() or "_" for x in word.split("_"))
    return output[0].lower() + output[1:]


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def format_array(arr: np.ndarray, style: str = "") -> str:
    msg = f"numpy.ndarray, shape={arr.shape}, dtype={arr.dtype}:\n"
    if style.endswith("_scale_offset_arr"):
        msg += f"shift_pre:\n{arr[:, 0]}\n"
        msg += f"scale:\n{arr[:, 1]}\n"
        msg += f"offset_scale:\n{arr[:, 2]}\n"
        msg += f"offset:\n{arr[:, 3]}\n"
        msg += f"shift_post:\n{arr[:, 4]}"
    else:
        msg += f"{arr}"
    return msg + "\n"


_RT = TypeVar("_RT")
_DecoratedFunc = TypeVar("_DecoratedFunc", bound=Callable[..., _RT])


def log_method_output(
    level: int = logging.DEBUG, logger: Optional[logging.Logger] = None
) -> Callable[[_DecoratedFunc], _DecoratedFunc]:
    def _log_method_output(func: _DecoratedFunc) -> _DecoratedFunc:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> _RT:
            try:
                logger = logger or self.logger
            except AttributeError:
                logger = logging.getLogger()

            out: _RT = func(self, *args, **kwargs)
            msg = f"{func.__name__} output:\n"
            if isinstance(out, np.ndarray):
                msg += format_array(out, func.__name__)
            else:
                msg += f"{out}\n"

            logger.log(level, msg)
            return out

        return cast(_DecoratedFunc, wrapper)

    return _log_method_output


class LoggingContext:
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
        close: bool = True,
    ) -> None:
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self) -> None:
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def quantize(
    arr: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: Union[type, "np.dtype"] = np.int8,
) -> np.ndarray:
    t = np.round(np.float32(arr) / np.float32(scale)).astype(np.int32) + zero_point
    return dtype(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max))


def dequantize(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.float32(arr.astype(np.int32) - np.int32(zero_point)) * np.float32(scale)

