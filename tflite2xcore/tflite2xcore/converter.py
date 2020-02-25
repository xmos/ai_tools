# Copyright (c) 2019, XMOS Ltd, All rights reserved

from tflite2xcore.graph_transformer import PassManager, PassPriority
from tflite2xcore import read_flatbuffer, write_flatbuffer
from tflite2xcore import transformation_passes as passes


def strip_model(model, *, remove_softmax=False):
    pass_mgr = PassManager(
        model,
        passes=[
            passes.RemoveQuantizerFloatInputPass(),
            passes.RemoveDequantizerFloatOutputPass(),
            passes.RemoveUnusedBuffersPass()
        ]
    )

    if remove_softmax:
        pass_mgr.register_pass(passes.RemoveSoftmaxOutputPass())

    pass_mgr.run_passes()
    model.description = 'TOCO converted + XMOS stripped.'


def add_float_input_output(model):
    pass_mgr = PassManager(
        model,
        passes=[
            passes.AddQuantizerFloatInputPass(),
            passes.AddDequantizerFloatOutputPass()
        ]
    )

    pass_mgr.run_passes()
    model.description = 'TOCO converted + XMOS stripped + float interface'

    # fix input/output buffers so built-in interpreter could run it
    assert len(model.subgraphs) == 1
    subgraph = model.subgraphs[0]
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    input_tensor = subgraph.inputs[0]
    output_tensor = subgraph.outputs[0]

    model.buffers.remove(input_tensor.buffer)
    model.buffers.remove(output_tensor.buffer)
    input_tensor.buffer = output_tensor.buffer
    model.buffers.insert(0, input_tensor.buffer)


def optimize_for_xcore(model, *, is_classifier, remove_softmax):
    pass_mgr = PassManager(
        model,
        passes=[
            passes.RemoveQuantizerFloatInputPass(),
            passes.RemoveDequantizerFloatOutputPass()
        ]
    )

    if is_classifier or remove_softmax:
        pass_mgr.register_pass(passes.RemoveSoftmaxOutputPass())

    if is_classifier:
        pass_mgr.register_pass(passes.AddArgMax16OutputPass())

    pass_mgr.register_pass(passes.ReplaceArgMax16Pass())
    pass_mgr.register_pass(passes.ReplaceDeepinDeepoutConv2DPass())
    pass_mgr.register_pass(passes.ReplaceShallowinDeepoutConv2DPass())
    pass_mgr.register_pass(passes.ReplaceSingleinDeepoutDepthwiseConv2DPass())
    pass_mgr.register_pass(passes.ReplaceMaxPool2DPass())
    pass_mgr.register_pass(passes.ReplaceMaxPool2D2x2Pass())
    pass_mgr.register_pass(passes.ReplaceAveragePool2D2x2Pass())
    pass_mgr.register_pass(passes.ReplaceAveragePool2DPass())
    pass_mgr.register_pass(passes.ReplaceGlobalAveragePool2DPass())
    pass_mgr.register_pass(passes.ReplaceFullyConnectedIntermediatePass())
    pass_mgr.register_pass(passes.ReplaceFullyConnectedOutputPass())

    pass_mgr.register_pass(passes.ReplaceReLUPass())
    pass_mgr.register_pass(passes.ReplaceReLU6Pass())
    pass_mgr.register_pass(passes.ReplaceTanhPass())
    pass_mgr.register_pass(passes.ReplaceLogisticPass())

    pass_mgr.register_pass(passes.RemoveUnusedBuffersPass())

    pass_mgr.run_passes()

    model.description = 'TOCO + XMOS converted.'


def parallelize_for_xcore(model, *, num_threads):
    pass_mgr = PassManager(
        model,
        passes=[
            passes.ParallelizeDIDOPass(num_threads=num_threads)
        ]
    )

    pass_mgr.run_passes()


def convert(tflite_input_path, tflite_output_path, *,
            is_classifier=False, remove_softmax=False, num_threads=1):
    model = read_flatbuffer(tflite_input_path)
    optimize_for_xcore(model, is_classifier=is_classifier, remove_softmax=remove_softmax)
    parallelize_for_xcore(model, num_threads=num_threads)
    write_flatbuffer(model, tflite_output_path)
