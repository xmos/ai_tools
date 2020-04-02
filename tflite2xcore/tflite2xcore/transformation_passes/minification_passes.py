# Copyright (c) 2020, XMOS Ltd, All rights reserved

from .transformation_passes import OperatorMatchingPass


class LegalizeOutputTensorNamePass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op):
            if len(op.outputs) == 1:
                return not op.outputs[0].name.startswith(f"{op.name}/output")

            for j, tensor in enumerate(op.outputs):
                candidate_name = f"{op.name}/output_{j}"
                if not tensor.name.startswith(candidate_name):
                    return True

        return False
