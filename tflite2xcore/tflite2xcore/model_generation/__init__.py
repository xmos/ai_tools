# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import tensorflow as tf
from typing import Union, Dict, Any, TypeVar
from typing_extensions import Protocol

from tflite2xcore.utils import TFLiteModel

Configuration = Dict[str, Any]

T_co = TypeVar("T_co", covariant=True)


class Hook(Protocol[T_co]):
    def __call__(self) -> T_co:
        ...


from .model_generators import ModelGenerator
