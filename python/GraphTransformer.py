# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq

from abc import ABC, abstractmethod


class PassPriority(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        return last_values[-1] + 1 if last_values else 0

    # TODO: change these to meaningful names
    HIGH = enum.auto()
    MEDIUM = enum.auto()
    LOW = enum.auto()


class OptimizationPass(ABC):
    def __init__(self, priority):
        assert isinstance(priority, PassPriority)
        self.priority = priority

    @abstractmethod
    def match(self, op):
        pass

    @abstractmethod
    def mutate(self, op):
        pass

    def run_subgraph(self, subgraph):
        # TODO: assert that argument is valid subgraph object
        keep_running = True
        while keep_running:
            for op in subgraph.operators:
                if self.match(op):
                    # TODO: log a debug message here
                    self.mutate(op)
                    break
            else:
                keep_running = False

    def run(self, model):
        # TODO: assert that argument is valid model object
        for subgraph in model.subgraphs:
            self.run_subgraph(subgraph)


class PassManager():
    def __init__(self, model=None, passes=[]):
        self._queue = []
        if model:
            self.register_model(model)
        for opt_pass in passes:
            self.register_pass(opt_pass)

    def register_model(self, model):
        # TODO: assert that argument is valid model object
        self._model = model

    def register_pass(self, opt_pass):
        assert isinstance(opt_pass, OptimizationPass)
        heapq.heappush(self._queue, (opt_pass.priority, opt_pass))

    def run_passes(self):
        while self._queue:
            _, opt_pass = heapq.heappop(self._queue)
            # TODO: log a debug message here
            opt_pass.run(self._model)


# TODO: move this to separate file, add tests
class RemoveQuantizerFloatInput(OptimizationPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        # TODO: check that this is compliant with model data structure
        if op.opcode == "QUANTIZE":
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if input_tensor in op.subgraph.inputs:
                return output_tensor.type == 'INT8' and input_tensor.type == 'FLOAT32'

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.remove(op.inputs[0])
        subgraph.inputs.append(op.outputs[0])
        subgraph.operators.remove(op)


# TODO: move this to separate file, add tests
class RemoveDequantizerFloatOutput(OptimizationPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        # TODO: check that this is compliant with model data structure
        if op.opcode == "DEQUANTIZE":
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if output_tensor in op.subgraph.outputs:
                return output_tensor.type == 'FLOAT32' and input_tensor.type == 'INT8'

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.remove(op.outputs[0])
        subgraph.outputs.append(op.inputs[0])
        subgraph.operators.remove(op)
