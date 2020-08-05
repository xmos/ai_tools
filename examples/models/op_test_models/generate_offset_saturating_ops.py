#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

from pathlib import Path
import op_test_models_common as common
import tensorflow as tf
import numpy as np

DEFAULT_PATH = Path(__file__).parent.joinpath("debug").resolve()
DEFAULT_NAME = "offset_saturating"


class OffsetSaturatingModel(common.OpTestDefaultModel):
    def get_case_layers(self, case):
        source_model = tf.keras.applications.MobileNet(
            input_shape=(128, 128, 3), alpha=0.25
        )
        layer_idx_map = {
            0: [1, 2, 3, 4],
            1: [8, 9, 10],
            2: [15, 16, 17],
            3: [21, 22, 23],
        }
        return [source_model.layers[idx] for idx in layer_idx_map[case]]

    def build_core_model(self, *, case=0, input_init):
        new_layers = self.get_case_layers(case)
        input_shape = new_layers[0].input_shape[1:]

        self.input_init = input_init
        self.core_model = tf.keras.models.Sequential(
            layers=[tf.keras.layers.InputLayer(input_shape), *new_layers]
        )

    def gen_test_data(self, batch=10, subset_len=10):
        super().gen_test_data(batch=batch, subset_len=subset_len)


class OpTestSpecificCaseParser(common.OpTestInitializerParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "--name", default=defaults["name"], help="Chosen test case number.",
        )
        self.add_argument(
            "--case",
            default=defaults["choices"][0],
            choices=defaults["choices"],
            type=int,
            help="Chosen test case number.",
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        args.name = f"{args.name}_{args.case}"
        args.path = args.path.joinpath(args.name)
        return args


def main(raw_args=None):
    parser = OpTestSpecificCaseParser(
        defaults={
            "name": DEFAULT_NAME,
            "path": DEFAULT_PATH,
            "choices": [0, 1, 2, 3],
            "inits": {"input_init": {"type": common.OpTestInitializers.UNIF}},
        }
    )
    args = parser.parse_args(raw_args)

    model = OffsetSaturatingModel(args.name, path=args.path)
    model.build(case=args.case, **args.inits)
    model.run()


if __name__ == "__main__":
    main()
