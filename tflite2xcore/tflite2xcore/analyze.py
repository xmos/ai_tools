# Copyright (c) 2020, XMOS Ltd, All rights reserved

from copy import copy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType
import tflite2xcore.xcore_interpreter as xcore_interpreter
from tflite2xcore import xlogging as logging


def calc_subgraph_mem_req(subgraph):
    operators = copy(subgraph.operators)
    coexisting = set(subgraph.inputs)
    op_mem_reqs = dict()
    while operators:
        op = operators.pop(0)
        needed_tensors = op.inputs + op.outputs
        coexisting.update(needed_tensors)

        op_mem_reqs[op.name] = {
            "buffer_tensors": {
                tensor.name: len(tensor.buffer.data)
                for tensor in coexisting
                if len(tensor.buffer.data)
            },
            "arena_tensors": {
                tensor.name: tensor.size
                for tensor in coexisting
                if not len(tensor.buffer.data)
            },
            "init": 500,  # TODO: this is a rough estimate
            "stack": 1000,  # TODO: this is a rough estimate
        }
        op_mem_reqs[op.name]["buffers"] = sum(
            v for v in op_mem_reqs[op.name]["buffer_tensors"].values()
        )
        op_mem_reqs[op.name]["arena"] = sum(
            v for v in op_mem_reqs[op.name]["arena_tensors"].values()
        )

        coexisting = {
            tensor
            for tensor in coexisting
            if (
                [consumer for consumer in tensor.consumers if consumer in operators]
                or tensor in subgraph.outputs
            )
        }

    return {
        "op_mem_reqs": op_mem_reqs,
        "buffers": sum(
            len(buffer.data)
            for buffer in subgraph.model.buffers
            if [owner for owner in buffer.owners if owner in subgraph.tensors]
        ),
        "arena": max(op_info["arena"] for op_info in op_mem_reqs.values()),
        "init": sum(op_info["init"] for op_info in op_mem_reqs.values()),
        "stack": max(op_info["stack"] for op_info in op_mem_reqs.values()),
    }


def analyze_model(model):
    analysis = {
        "subgraphs": [calc_subgraph_mem_req(subgraph) for subgraph in model.subgraphs]
    }
    analysis["buffers"] = sum(len(buffer.data) for buffer in model.buffers)
    analysis["arena"] = max(
        subgraph_info["arena"] for subgraph_info in analysis["subgraphs"]
    )
    analysis["init"] = sum(
        subgraph_info["init"] for subgraph_info in analysis["subgraphs"]
    )
    analysis["stack"] = max(
        subgraph_info["stack"] for subgraph_info in analysis["subgraphs"]
    )

    return analysis


# TODO: remove this someday since analysis should not rely on an interpreter
#       however, currently the interpreter is the only method to determine the
#       size of the tensor arena
def calc_arena_sizes(model_content):
    interpreter = xcore_interpreter.XCOREInterpreter(model_content=model_content)
    return interpreter.tensor_arena_size, interpreter.xcore_heap_size


def calc_weight_and_bias_fetch_sizes(model_content):
    max_weights_size = 0
    max_bias_size = 0
    model = XCOREModel.deserialize(model_content)
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if "mem" in op.custom_options:
                max_weights_size = max(max_weights_size, op.custom_options["mem"][0])
                max_bias_size = max(max_bias_size, op.custom_options["mem"][1])

    return max_weights_size, max_bias_size


def print_report(tflite_output_path):
    indent = " " * 2

    with open(tflite_output_path, "rb") as fd:
        model_content = fd.read()
        model_size = len(model_content)
        try:
            tensor_arena_size, xcore_heap_size = calc_arena_sizes(model_content)
            max_weights_size, max_bias_size = calc_weight_and_bias_fetch_sizes(
                model_content
            )
            print(f"Model size: {model_size} (bytes)")
            print()
            print("Model stored in RAM")
            print(f"{indent}Tensor arena size: {tensor_arena_size} (bytes)")
            print(f"{indent}xCORE heap size: {xcore_heap_size} (bytes)")
            print()
            print(
                f"{indent}Total RAM required: {model_size + tensor_arena_size + xcore_heap_size} (bytes)"
            )
            print()
            print("Model stored in external memory (Flash or LPDDR)")
            print(f"{indent}Tensor arena size: {tensor_arena_size} (bytes)")
            print(
                f"  xCORE heap size: {xcore_heap_size + max_weights_size + max_bias_size} (bytes)"
            )
            print()
            print(f"{indent}Total RAM required: {xcore_heap_size + tensor_arena_size}")
            print(f"{indent}Total external memory required: {model_size}")
            print()
        except RuntimeError as e:
            prefix = "Didn't find op for builtin opcode "
            msg = e.args[0]
            if msg.startswith(prefix):
                op_details = msg.split("\n", 1)[0][len(prefix) :]
                logging.getLogger().warning(
                    f"Arena size calculation failed because of unknown op in the interpreter: {op_details}"
                )
            else:
                raise
