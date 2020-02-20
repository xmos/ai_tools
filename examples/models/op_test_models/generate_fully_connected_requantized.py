#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tflite2xcore.converter as xcore_conv
from tflite2xcore import read_flatbuffer, write_flatbuffer
import op_test_models_common as common
from generate_fully_connected import FullyConnected
from generate_fully_connected import (DEFAULT_OUTPUT_DIM, DEFAULT_INPUT_DIM,
                                      DEFAULT_EPOCHS, DEFAULT_BS)
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'fully_connected_requantized').resolve()


class FullyConnectedRequantized(FullyConnected):
    def to_tf_xcore(self):
        assert 'model_quant' in self.models
        self.models['model_xcore'] = str(self.models['models_dir'] / 'model_xcore.tflite')
        model = read_flatbuffer(str(self.models['model_quant']))

        # NOTE: since the output softmax is not removed during the first
        # conversion, ReplaceFullyConnectedIntermediatePass will
        # match and insert the requantization. Then, the softmax can be removed.
        xcore_conv.optimize_for_xcore(model,
                                      is_classifier=False,
                                      remove_softmax=False)
        xcore_conv.strip_model(model, remove_softmax=True)

        model = write_flatbuffer(model, self.models['model_xcore'])


def main(path=DEFAULT_PATH, *,
         input_dim=DEFAULT_INPUT_DIM,
         output_dim=DEFAULT_OUTPUT_DIM,
         train_new_model=False,
         bias_init=common.DEFAULT_CONST_INIT,
         weight_init=common.DEFAULT_UNIF_INIT):
    kwargs = {
        'name': 'fc_deepin_anyout_requantized',
        'path': path if path else DEFAULT_PATH
    }
    common.run_main_fc(model=FullyConnectedRequantized(**kwargs),
                       train_new_model=train_new_model,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       bias_init=bias_init,
                       weight_init=weight_init,
                       batch_size=None,
                       epochs=None)


if __name__ == "__main__":
    parser = common.OpTestFCParser(defaults={
        'path': DEFAULT_PATH,
        'input_dim': DEFAULT_INPUT_DIM,
        'output_dim': DEFAULT_OUTPUT_DIM
    })
    args = parser.parse_args()
    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         input_dim=args.input_dim,
         output_dim=args.output_dim,
         train_new_model=args.train_model)
