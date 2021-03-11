# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from .transformation_passes import OperatorMatchingPass


class LegalizeOperatorOutputTensorNamePass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op):
            if len(op.outputs) == 1:
                return not op.outputs[0].name.startswith(f"{op.name}/output")

            for j, tensor in enumerate(op.outputs):
                candidate_name = f"{op.name}/output_{j}"
                if not tensor.name.startswith(candidate_name):
                    return True

        return False

    def __mutate_tensor_name(self, tensor, candidate_name):
        subgraph = tensor.subgraph
        if tensor.name != candidate_name:
            unique_name = subgraph.make_unique_tensor_name(candidate_name)

            if unique_name is not candidate_name:
                self.logger.warning(
                    f"candidate_name {candidate_name} is already used by "
                    f"tensor {subgraph.tensors.index(tensor)}, "
                    f"defaulting to {unique_name}"
                )

            tensor.name = unique_name

    def mutate(self, op):
        if len(op.outputs) == 1:
            self.__mutate_tensor_name(op.outputs[0], f"{op.name}/output")
        else:
            for j, tensor in enumerate(op.outputs):
                self.__mutate_tensor_name(tensor, f"{op.name}/output_{j}")
