# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from .flatbuffers_io import (
    read_flatbuffer,
    write_flatbuffer,
    serialize_model,
    deserialize_model,
    FlexbufferParser
)
from .converters import create_dict_from_model
