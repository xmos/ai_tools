# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pathlib

from tflite2xcore.pass_manager import PassManager
from tflite2xcore.serialization import read_flatbuffer, write_flatbuffer
from tflite2xcore import transformation_passes as passes


def strip_model(model, *, remove_softmax=False, debug=False, legalize_op_versions=True):
    pass_mgr = PassManager(
        model,
        passes=[
            passes.CanonicalizeQuantizedInputPass(),
            passes.CanonicalizeQuantizedOutputPass(),
        ],
        debug=debug,
    )

    if remove_softmax:
        pass_mgr.register_pass(passes.RemoveSoftmaxOutputPass())

    if legalize_op_versions:
        pass_mgr.register_pass(passes.LegalizeQuantizeVersionPass())

    pass_mgr.register_pass(passes.RemoveDanglingTensorsPass())
    pass_mgr.register_pass(passes.RemoveUnusedBuffersPass())

    pass_mgr.run_passes()
    model.description = model.description + " + XMOS stripped."


def add_float_input_output(model, debug=False):
    pass_mgr = PassManager(
        model,
        passes=[passes.LegalizeFloatInputPass(), passes.LegalizeFloatOutputPass()],
        debug=debug,
    )

    pass_mgr.run_passes()
    model.description = model.description + " float interface."

    # fix input/output buffers so built-in interpreter could run it
    assert len(model.subgraphs) == 1
    subgraph = model.subgraphs[0]
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    input_tensor = subgraph.inputs[0]
    output_tensor = subgraph.outputs[0]

    assert len(input_tensor.buffer.owners) == 1
    input_tensor.buffer.owners = []
    model.buffers.remove(input_tensor.buffer)

    input_tensor.buffer = output_tensor.buffer
    input_tensor.buffer.owners.append(input_tensor)
    model.buffers.remove(input_tensor.buffer)
    model.buffers.insert(0, input_tensor.buffer)


def optimize_for_xcore(
    model,
    *,
    remove_softmax=False,
    cleanup=True,
    minification=False,
    num_threads=None,
    intermediates_path=None,
    debug=False
):
    # NOTE: the order of the passes is mostly strict
    pass_mgr = PassManager(
        model,
        passes=[
            # TODO: these are actually canonicalization passes
            passes.CanonicalizeQuantizedInputPass(),
            passes.CanonicalizeQuantizedOutputPass(),
            passes.SplitPaddingPass(),
        ],
        keep_intermediates=bool(intermediates_path),
        debug=debug,
    )

    # TODO: remove this
    if remove_softmax:
        pass_mgr.register_pass(passes.RemoveSoftmaxOutputPass())

    pass_mgr.register_pass(passes.ReplaceReLUPass())
    pass_mgr.register_pass(passes.ReplaceReLU6Pass())
    pass_mgr.register_pass(passes.ReplaceTanhPass())
    pass_mgr.register_pass(passes.ReplaceLogisticPass())

    pass_mgr.register_pass(passes.Replace1x1Conv2dPass())
    pass_mgr.register_pass(passes.ReplaceDepthwiseConv2dPass())
    pass_mgr.register_pass(passes.ReplaceDeepConv2dPass())

    pass_mgr.register_pass(passes.ReplaceMaxPool2D2x2Pass())
    pass_mgr.register_pass(passes.ReplaceMaxPool2DPass())
    pass_mgr.register_pass(passes.ReplaceAveragePool2D2x2Pass())
    pass_mgr.register_pass(passes.ReplaceAveragePool2DPass())
    pass_mgr.register_pass(passes.ReplaceGlobalAveragePool2DPass())

    pass_mgr.register_pass(passes.ReplaceFullyConnectedPass())

    pass_mgr.register_pass(passes.LegalizeXCLookupTablePass())
    pass_mgr.register_pass(passes.LegalizeXCFullyConnectedPass())
    pass_mgr.register_pass(passes.LegalizeXCDepthwiseConvPass())
    pass_mgr.register_pass(passes.LegalizeXC1x1ConvPass())
    pass_mgr.register_pass(passes.LegalizeXCDeepConvPass())

    pass_mgr.register_pass(passes.FuseConv2dPaddingPass())
    pass_mgr.register_pass(passes.FuseConsecutivePadsPass())

    if num_threads:
        pass_mgr.register_pass(
            passes.ParallelizeDeepConv2dPass(num_threads=num_threads)
        )

    if cleanup:
        pass_mgr.register_pass(passes.RemoveXCOREWeightBiasOperatorQuantInfo())
        pass_mgr.register_pass(passes.RemoveDanglingTensorsPass())
        pass_mgr.register_pass(passes.RemoveUnusedBuffersPass())

    # TODO: this is actually a canonicalization pass
    pass_mgr.register_pass(passes.LegalizeOperatorOutputTensorNamePass())
    pass_mgr.register_pass(passes.LegalizeQuantizeVersionPass())

    if minification:
        pass_mgr.register_pass(passes.MinifyQuantInfoPass())
        pass_mgr.register_pass(passes.MinifyTensorNamesPass())

    pass_mgr.run_passes()
    model.sanity_check()

    model.description = model.description + " + XMOS optimized."

    if pass_mgr.keep_intermediates:
        pass_mgr.save_intermediates(intermediates_path)


def convert(
    tflite_input_path,
    tflite_output_path,
    *,
    remove_softmax=False,
    num_threads=None,
    minification=False,
    intermediates_path=None,
    debug=False
):
    model = read_flatbuffer(tflite_input_path)
    optimize_for_xcore(
        model,
        remove_softmax=remove_softmax,
        minification=minification,
        num_threads=num_threads,
        intermediates_path=intermediates_path,
        debug=debug,
    )
    write_flatbuffer(model, tflite_output_path)
