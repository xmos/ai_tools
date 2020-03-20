#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tflite2xcore.converter as xcore_conv
from tflite2xcore.serialization.flatbuffers_io import serialize_model, deserialize_model
import op_test_models_common as common

from generate_fully_connected import (
    DEFAULT_OUTPUT_DIM, DEFAULT_INPUT_DIM,
    DEFAULT_EPOCHS, DEFAULT_BS
)
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'fully_connected_requantized').resolve()


class FullyConnectedRequantized(common.OpTestDefaultFCModel):
    def convert_to_xcore(self, **converter_args):
        # NOTE: since the output softmax is not removed during the first
        # conversion, ReplaceFullyConnectedIntermediatePass will
        # match and insert the requantization. Then, the softmax can be removed.
        super().convert_to_xcore(remove_softmax=False, **converter_args)
        model = deserialize_model(self.buffers['model_xcore'])
        xcore_conv.strip_model(model, remove_softmax=True)
        self.buffers['model_xcore'] = serialize_model(model)


def main(raw_args=None):
    parser = common.OpTestFCParser(defaults={
        'path': DEFAULT_PATH,
        'input_dim': DEFAULT_INPUT_DIM,
        'output_dim': DEFAULT_OUTPUT_DIM,
        'batch_size': DEFAULT_BS,
        'epochs': DEFAULT_EPOCHS,
        'inits': {
            'weight_init': {'type': common.OpTestInitializers.UNIF},
            'bias_init': {'type': common.OpTestInitializers.CONST}
        }
    })
    args = parser.parse_args(raw_args)

    model = FullyConnectedRequantized('fc_deepin_anyout_requantized', args.path)
    if args.train_model:
        model.build_and_train(args.input_dim, args.output_dim,
                              args.batch_size, args.epochs,
                              **args.inits)
    else:
        model.load_core_model(args.output_dim)
    model.convert_and_save()


if __name__ == "__main__":
    main()
