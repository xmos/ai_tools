# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
from copy import copy

from tflite2xcore.xcore_model import XCOREModel


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

    return analysis


# TODO: remove this someday since analysis should not rely on an interpreter
#       however, currently the interpreter is the only method to determine the
#       size of the tensor arena
def calc_arena_size(model_content):
    try:
        from xcore_interpreters import XCOREInterpreter

        interpreter = XCOREInterpreter(model_content=model_content)
        logger = logging.getLogger("tensor_arena_allocations")
        [logger.info(line) for line in interpreter.get_allocations().split("\n")]
        return interpreter.tensor_arena_size
    except RuntimeError as e:
        print("Runtime Error: Failed calculating tensor arena size.")
        print(str(e))
        return None


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
            tensor_arena_size = calc_arena_size(model_content)
            max_weights_size, max_bias_size = calc_weight_and_bias_fetch_sizes(
                model_content
            )
            print(f"Model size: {model_size} (bytes)")
            print()
            if tensor_arena_size:
                ram_used = model_size + tensor_arena_size
                print("Model stored in RAM")
                print(f"{indent}Tensor arena size: {tensor_arena_size} (bytes)")
                print()
                print(f"{indent}Total RAM required: {ram_used} (bytes)")
                print()
                if max_weights_size and max_bias_size:
                    print("Model stored in external memory (Flash or LPDDR)")
                    tensor_arena_size += max_weights_size + max_bias_size
                    print(f"{indent}Tensor arena size: {tensor_arena_size} (bytes)")
                    print()
                    print(f"{indent}Total RAM required: {tensor_arena_size}")
                    print(f"{indent}Total external memory required: {model_size}")
                    print()
            else:
                print("Unable to determine model memory requirements.")
        except RuntimeError as e:
            prefix = "Didn't find op for builtin opcode "
            msg = e.args[0]
            if msg.startswith(prefix):
                op_details = msg.split("\n", 1)[0][len(prefix) :]
                logging.warning(
                    f"Arena size calculation failed because of unknown op in the interpreter: {op_details}"
                )
            else:
                raise
