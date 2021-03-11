# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import os
import re
import random
import argparse
import logging
import numpy as np
import tensorflow as tf
from math import log2, ceil
from functools import wraps
from types import TracebackType
from typing import (
    Union,
    Optional,
    Dict,
    Any,
    TypeVar,
    Callable,
    cast,
    Type,
    NamedTuple,
    Iterator,
    List,
    Tuple,
)

# -----------------------------------------------------------------------------
#                          WIDELY USED TYPES
# -----------------------------------------------------------------------------


class QuantizationTuple(NamedTuple):
    scale: float
    zero_point: int


TFLiteModel = Union[bytes, bytearray]
PaddingTuple = Tuple[Tuple[int, int], ...]
ShapeTuple = Tuple[int, ...]


# -----------------------------------------------------------------------------
#                          XCORE MAGIC NUMBERS
# -----------------------------------------------------------------------------

ACC_PERIOD_INT8 = 16

WORD_SIZE_BYTES = 4
WORD_SIZE_BITS = WORD_SIZE_BYTES * 8
VECTOR_SIZE_WORDS = 8
VECTOR_SIZE_BYTES = VECTOR_SIZE_WORDS * 4
VECTOR_SIZE_BITS = VECTOR_SIZE_BYTES * 8

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
    try:
        default_log_level = os.environ["TF_CPP_MIN_LOG_LEVEL"]
        if not verbose:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        gpus = tf.config.experimental.list_physical_devices("GPU")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = default_log_level
    except KeyError:
        gpus = tf.config.experimental.list_physical_devices("GPU")

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


# -----------------------------------------------------------------------------
#                          BINARY OPERATION HELPERS
# -----------------------------------------------------------------------------


def unpack_bits(arr: np.ndarray) -> np.ndarray:
    assert arr.dtype == np.int32
    unpacked_shape = (*arr.shape[:-1], arr.shape[-1] * WORD_SIZE_BITS)
    return np.unpackbits(  # pylint: disable=no-member
        np.frombuffer(arr.tobytes(), dtype=np.uint8)
    ).reshape(unpacked_shape)


def xor_popcount(a: np.ndarray, b: np.ndarray) -> int:
    assert a.dtype == b.dtype == np.int32
    return np.count_nonzero(unpack_bits(np.bitwise_xor(a, b)))  # type: ignore


def clrsb(a: int, bitwidth: int = 32) -> int:
    """ counts leading redundant sign bits """
    return bitwidth - ceil(log2(abs(a))) - 1


# -----------------------------------------------------------------------------
#                          QUANTIZATION HELPERS
# -----------------------------------------------------------------------------


def quantize(
    arr: np.ndarray,
    scale: float,
    zero_point: int,
    dtype: Union[type, "np.dtype"] = np.int8,
) -> np.ndarray:
    t = np.round(np.float32(arr) / np.float32(scale)).astype(np.int32) + zero_point
    return np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def dequantize(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return np.float32(arr.astype(np.int32) - np.int32(zero_point)) * np.float32(scale)


# -----------------------------------------------------------------------------
#                       MODEL CONVERSION AND EVALUATION HELPERS
# -----------------------------------------------------------------------------


def quantize_converter(
    converter: tf.lite.TFLiteConverter,
    representative_data: Union[tf.Tensor, np.ndarray],
    *,
    show_progress_step: int = 0,
) -> None:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    x_train_ds = tf.data.Dataset.from_tensor_slices(representative_data).batch(1)

    def representative_data_gen() -> Iterator[List[tf.Tensor]]:
        for j, input_value in enumerate(x_train_ds.take(representative_data.shape[0])):
            if show_progress_step and (j + 1) % show_progress_step == 0:
                logging.info(
                    "Converter quantization processed examples "
                    f"{j+1:6d}/{representative_data.shape[0]}"
                )
            yield [input_value]

    converter.representative_dataset = representative_data_gen


def apply_interpreter_to_examples(
    interpreter: tf.lite.Interpreter,
    examples: Union[tf.Tensor, np.ndarray],
    *,
    interpreter_input_ind: Optional[int] = None,
    interpreter_output_ind: Optional[int] = None,
    show_progress_step: int = 0,
    show_pid: bool = False,
) -> np.ndarray:
    interpreter.allocate_tensors()
    if interpreter_input_ind is None:
        interpreter_input_ind = interpreter.get_input_details()[0]["index"]
    if interpreter_output_ind is None:
        interpreter_output_ind = interpreter.get_output_details()[0]["index"]

    outputs = []
    for j, x in enumerate(examples):
        if show_progress_step and (j + 1) % show_progress_step == 0:
            if show_pid:
                logging.info(
                    f"(PID {os.getpid()}) Evaluated examples {j+1:6d}/{examples.shape[0]}"
                )
            else:
                logging.info(f"Evaluated examples {j+1:6d}/{examples.shape[0]}")
        interpreter.set_tensor(interpreter_input_ind, np.expand_dims(x, 0))
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter_output_ind)
        outputs.append(y)

    return np.vstack(outputs) if isinstance(examples, np.ndarray) else outputs


def quantize_keras_model(
    model: tf.keras.Model,
    representative_data: Union[tf.Tensor, np.ndarray],
    show_progress_step: int = 0,
) -> TFLiteModel:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantize_converter(
        converter, representative_data, show_progress_step=show_progress_step
    )
    return converter.convert()  # type: ignore


# -----------------------------------------------------------------------------
#                       SHAPE COMPUTATION HELPERS
# -----------------------------------------------------------------------------


def _calculate_valid_output_size(in_size: int, stride: int, k_dim: int) -> int:
    assert in_size >= k_dim
    return ceil((in_size - k_dim + 1) / stride)


def calculate_valid_output_size(
    input_size: ShapeTuple, strides: ShapeTuple, kernel_size: ShapeTuple
) -> ShapeTuple:
    return tuple(
        _calculate_valid_output_size(*t) for t in zip(input_size, strides, kernel_size)
    )


def _calculate_same_output_size(in_size: int, stride: int) -> int:
    return ceil(in_size / stride)


def calculate_same_output_size(
    input_size: ShapeTuple, strides: ShapeTuple
) -> ShapeTuple:
    return tuple(_calculate_same_output_size(*t) for t in zip(input_size, strides))


def calculate_same_padding(
    input_size: ShapeTuple, strides: ShapeTuple, kernel_size: ShapeTuple
) -> PaddingTuple:
    def calc_axis_pad(in_size: int, stride: int, k_dim: int) -> Tuple[int, int]:
        out_size = _calculate_same_output_size(in_size, stride)
        total_pad = max((out_size - 1) * stride + k_dim - in_size, 0)
        pad_start = total_pad // 2
        return (pad_start, total_pad - pad_start)

    return tuple(calc_axis_pad(*t) for t in zip(input_size, strides, kernel_size))


def get_bitpacked_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    channels = shape[-1]
    assert channels % WORD_SIZE_BITS == 0
    return (*shape[:-1], channels // WORD_SIZE_BITS)


def get_unpacked_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return (*shape[:-1], shape[-1] * WORD_SIZE_BITS)
