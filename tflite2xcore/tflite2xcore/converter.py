# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pathlib

from tflite2xcore.pass_manager import PassManager
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore import transformation_passes as passes


class CleanupManager(PassManager):
    def __init__(self, model=None, **kwargs):
        super().__init__(
            model,
            passes=[
                passes.EliminateDeadOperatorsPass(),
                passes.EliminateDeadTensorsPass(),
                passes.EliminateDeadBuffersPass(),
            ],
            **kwargs
        )


class InputOutputCanonicalizationManager(PassManager):
    def __init__(self, model=None, **kwargs):
        super().__init__(
            model,
            passes=[
                passes.CanonicalizeQuantizedInputPass(),
                passes.CanonicalizeQuantizedOutputPass(),
            ],
            **kwargs
        )


def strip_model(model, *, debug=False, legalize_op_versions=True):
    pass_mgr = InputOutputCanonicalizationManager(model, debug=debug)

    if legalize_op_versions:
        pass_mgr.register_pass(passes.LegalizeQuantizeVersionPass())

    pass_mgr.register_passes(CleanupManager())

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
    cleanup=True,
    minification=False,
    num_threads=1,
    intermediates_path=None,
    debug=False,
    ignore_input_alignment=False
):
    # NOTE: the order of the passes is mostly strict
    pass_mgr = InputOutputCanonicalizationManager(
        model, keep_intermediates=bool(intermediates_path), debug=debug,
    )

    pass_mgr.register_pass(passes.CanonicalizeReshapePass())
    pass_mgr.register_pass(passes.RemoveFlattenReshapePass())

    # canonicalize convolutions
    pass_mgr.register_pass(passes.CanonicalizeSingleinDepthwiseConv2DPass())
    pass_mgr.register_pass(passes.LegalizeSingleinConv2DPass())

    # canonicalize word alignment
    pass_mgr.register_pass(passes.CanonicalizeConv2DInputChannels())

    # word alignment canonicalization introduces new pads, so first fuse then split
    pass_mgr.register_pass(passes.FuseConsecutivePadsPass())

    # Split batch/channel-wise padding from spacial padding - allows fusing of spacial padding later
    pass_mgr.register_pass(passes.SplitPaddingPass())

    # need to cleanup after the first round of canonicalization
    pass_mgr.register_passes(CleanupManager())

    pass_mgr.register_pass(passes.ReplaceReLUPass())
    pass_mgr.register_pass(passes.ReplaceReLU6Pass())
    pass_mgr.register_pass(passes.ReplaceTanhPass())
    pass_mgr.register_pass(passes.ReplaceLogisticPass())

    pass_mgr.register_pass(passes.Replace1x1Conv2dPass())
    pass_mgr.register_pass(passes.ReplaceShallowinConv2dPass())
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
    pass_mgr.register_pass(passes.LegalizeXC1x1ConvPass())
    pass_mgr.register_pass(passes.LegalizeXCShallowinConvPass())
    pass_mgr.register_pass(passes.LegalizeXCDepthwiseConvPass())
    pass_mgr.register_pass(passes.LegalizeXCDeepConvPass())

    # Fuse spacial padding with conv2d
    pass_mgr.register_pass(passes.FuseConv2dPaddingPass())

    if ignore_input_alignment:
        pass_mgr.register_pass(passes.RemovePaddingInputPass())

    pass_mgr.register_pass(passes.FuseConsecutivePadsPass())

    pass_mgr.register_pass(
        passes.ParallelizeFullyConnectedPass(num_threads=num_threads)
    )
    pass_mgr.register_pass(passes.ParallelizeRequant16To8Pass(num_threads=num_threads))
    pass_mgr.register_pass(passes.ParallelizeConv2dPass(num_threads=num_threads))
    pass_mgr.register_pass(passes.ParallelizePooling2DPass(num_threads=num_threads))
    pass_mgr.register_pass(
        passes.ParallelizeGlobalAveragePool2DPass(num_threads=num_threads)
    )

    # NOTE: scratch memory passes must be registered after parallelization passes
    pass_mgr.register_pass(passes.ScratchMemoryConv2dPass())
    pass_mgr.register_pass(passes.ScratchMemoryConv2d1x1Pass())
    pass_mgr.register_pass(passes.ScratchMemoryFullyConnectedPass())

    if cleanup:
        pass_mgr.register_passes(CleanupManager())

    # TODO: this is actually a canonicalization pass
    pass_mgr.register_pass(passes.LegalizeOperatorOutputTensorNamePass())
    pass_mgr.register_pass(passes.LegalizeQuantizeVersionPass())

    if minification:
        pass_mgr.register_pass(passes.MinifyQuantInfoPass())
        pass_mgr.register_pass(passes.MinifyTensorNamesPass())

    try:
        pass_mgr.run_passes()
    finally:
        if pass_mgr.keep_intermediates:
            pass_mgr.save_intermediates(intermediates_path)

    model.sanity_check()

    model.description = model.description + " + XMOS optimized."


def convert(
    tflite_input_path,
    tflite_output_path,
    *,
    num_threads=None,
    minification=False,
    intermediates_path=None,
    debug=False
):
    num_threads = num_threads or 1
    model = XCOREModel.read_flatbuffer(tflite_input_path)
    optimize_for_xcore(
        model,
        minification=minification,
        num_threads=num_threads,
        intermediates_path=intermediates_path,
        debug=debug,
    )
    model.write_flatbuffer(tflite_output_path)
