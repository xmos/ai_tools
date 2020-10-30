# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import numpy as np
from typing import TYPE_CHECKING, Dict, Union, Any

from tflite2xcore.utils import camel_to_snake, snake_to_camel

from .flexbuffers import FlexbufferBuilder
from . import (
    schema_py_generated as schema,
    Buffer,
    Subgraph,
    Tensor,
    Operator,
    Metadata,
    ActivationFunctionType,
    Padding,
    OperatorCode,
    QuantizationDetails,
    BuiltinOpCodes,
    BuiltinOptions,
)

if TYPE_CHECKING:
    from . import XCOREModel


def create_dict_from_operator_code(
    operator_code: OperatorCode,
) -> Dict[str, Union[str, int]]:
    operator_code_dict: Dict[str, Union[str, int]] = {"version": operator_code.version}

    if operator_code.code in BuiltinOpCodes:
        operator_code_dict["builtin_code"] = operator_code.name
    else:
        operator_code_dict["builtin_code"] = BuiltinOpCodes.CUSTOM.name
        operator_code_dict["custom_code"] = operator_code.name

    return operator_code_dict


def create_dict_from_operator(operator: Operator,) -> Dict[str, Any]:
    tensors = operator.subgraph.tensors
    operator_codes = operator.subgraph.model.operator_codes

    operator_dict = {
        "opcode_index": operator_codes.index(operator.operator_code),
        "inputs": [tensors.index(input_tensor) for input_tensor in operator.inputs],
        "outputs": [tensors.index(input_tensor) for input_tensor in operator.outputs],
        "custom_options_format": "FLEXBUFFERS",
    }

    if operator.builtin_options:
        operator_dict["builtin_options"] = operator.builtin_options

    if operator.custom_options:
        fbb = FlexbufferBuilder(operator.custom_options)
        operator_dict["custom_options"] = fbb.get_bytes()

    return operator_dict


def create_dict_from_tensor(
    tensor: Tensor, *, extended: bool = False
) -> Dict[str, Any]:
    subgraph = tensor.subgraph
    buffers = subgraph.model.buffers

    tensor_dict = {
        "name": tensor.name,
        "type": tensor.type.name,
        "shape": tensor.shape,
        "buffer": buffers.index(tensor.buffer),
    }

    if tensor.quantization:
        tensor_dict["quantization"] = tensor.quantization

    if extended:
        operators = subgraph.operators
        tensor_dict["consumers"] = sorted(operators.index(t) for t in tensor.consumers)
        tensor_dict["producers"] = sorted(operators.index(t) for t in tensor.producers)

    return tensor_dict


def create_dict_from_subgraph(
    subgraph: Subgraph, *, extended: bool = False
) -> Dict[str, Any]:
    tensors = subgraph.tensors

    subgraph_dict = {
        "tensors": [
            create_dict_from_tensor(tensor, extended=extended) for tensor in tensors
        ],
        "inputs": [tensors.index(input_tensor) for input_tensor in subgraph.inputs],
        "outputs": [tensors.index(output_tensor) for output_tensor in subgraph.outputs],
        "operators": [
            create_dict_from_operator(operator) for operator in subgraph.operators
        ],
    }

    if subgraph.name:
        subgraph_dict["name"] = subgraph.name

    return subgraph_dict


def create_dict_from_buffer(
    buffer: Buffer[Any], *, extended: bool = False
) -> Dict[str, Any]:
    buffer_dict: Dict[str, Any] = {
        "data": buffer.data
    } if buffer.data is not None else {}

    if extended:
        owners_dict: Dict[Union[int, str], Any] = dict()
        model = buffer.model

        # track down and tally all owners
        for owner in buffer.owners:
            if owner in model.metadata:
                metadata_owners = owners_dict.setdefault("metadata", [])
                metadata_owners.append(owner.name)
            else:  # owner is a tensor
                subgraph = owner.subgraph
                owner_idx = model.subgraphs.index(subgraph)
                owners_in_subgraph = owners_dict.setdefault(owner_idx, [])
                owners_in_subgraph.append(subgraph.tensors.index(owner))

        # sort the ordering
        owners_dict = dict(sorted(owners_dict.items()))
        for subgraph_idx in owners_dict:
            owners_dict[subgraph_idx].sort()

        buffer_dict["owners"] = owners_dict

    return buffer_dict


def create_dict_from_metadata(metadata: Metadata) -> Dict[str, Union[int, str, None]]:
    return {
        "name": metadata.name,
        "buffer": metadata.model.buffers.index(metadata.buffer),
    }


def create_dict_from_model(
    model: "XCOREModel", *, extended: bool = False
) -> Dict[str, Any]:
    return {
        "version": model.version,
        "description": model.description,
        "metadata": [
            create_dict_from_metadata(metadata) for metadata in model.metadata
        ],
        "buffers": [
            create_dict_from_buffer(buffer, extended=extended)
            for buffer in model.buffers
        ],
        "subgraphs": [
            create_dict_from_subgraph(subgraph, extended=extended)
            for subgraph in model.subgraphs
        ],
        "operator_codes": [
            create_dict_from_operator_code(operator_code)
            for operator_code in model.operator_codes
        ],
    }


def builtin_options_to_dict(builtin_options: Any) -> Dict[str, Any]:
    dict_ = {camel_to_snake(k): v for k, v in vars(builtin_options).items()}
    if "fused_activation_function" in dict_:
        dict_["fused_activation_function"] = ActivationFunctionType(
            dict_["fused_activation_function"]
        )
    if "padding" in dict_:
        dict_["padding"] = Padding(dict_["padding"])

    return dict_


def dict_to_builtin_options(type_: int, dict_: Dict[str, Any]) -> Any:
    class_identifier = BuiltinOptions(type_).name + "T"

    builtin_class = getattr(schema, class_identifier)
    builtin_options = builtin_class()

    for k, v in dict_.items():
        if k in ["fused_activation_function", "padding"]:
            # enum to value
            v = v.value

        setattr(builtin_options, snake_to_camel(k), v)

    return builtin_options


def quantization_to_dict(
    quantization: schema.QuantizationParametersT,
) -> Dict[str, Any]:
    def value_map(k: str, v: Any) -> Any:
        if k == "detailsType":
            v = QuantizationDetails(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        return v

    return {
        camel_to_snake(k): value_map(k, v)
        for k, v in vars(quantization).items()
        if v is not None
    }


def dict_to_quantization(dict_: Dict[str, Any]) -> schema.QuantizationParametersT:
    quantization: schema.QuantizationParametersT = schema.QuantizationParametersT()  # type: ignore

    for k, v in dict_.items():
        if k == "details_type":
            # enum to value
            v = v.value

        setattr(quantization, snake_to_camel(k), v)

    return quantization
