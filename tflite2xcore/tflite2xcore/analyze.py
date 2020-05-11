# Copyright (c) 2020, XMOS Ltd, All rights reserved

from copy import copy

from tflite2xcore.xcore_schema import TensorType
import tflite2xcore.xcore_interpreter as xcore_interpreter


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


def calc_arena_sizes(model_content):
    interpreter = xcore_interpreter.XCOREInterpreter(model_content=model_content)
    return interpreter.tensor_arena_size, interpreter.xcore_heap_size
