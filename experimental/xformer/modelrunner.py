# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import os
import tempfile
import sys
import subprocess
import numpy as np
import pathlib
import argparse

import tensorflow

from tflm_interpreter import TFLMInterpreter

XFORMER2_PATH = (pathlib.Path(__file__).resolve().parents[0] / "bazel-bin" /
                 "xcore-opt")


def get_xformed_model(model):
    with tempfile.TemporaryDirectory(suffix=str(os.getpid())) as dirname:
        input_path = pathlib.Path(dirname) / "input.tflite"

        with open(pathlib.Path(input_path).resolve(), "wb") as fd:
            fd.write(model)

        output_path = pathlib.Path(dirname) / "output.tflite"
        cmd = [str(XFORMER2_PATH), str(input_path), "-o", str(output_path)]
        p = subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           check=True)
        print(p.stdout)

        with open(pathlib.Path(output_path).resolve(), "rb") as fd:
            bits = bytes(fd.read())
    return bits


def test_inference(args):
    model = args.model

    with open(model, "rb") as fd:
        model_content = fd.read()

    interpreter = tensorflow.lite.Interpreter(
        model_content=model_content,
        experimental_op_resolver_type=tensorflow.lite.experimental.
        OpResolverType.BUILTIN_REF)
    # interpreter = tensorflow.lite.Interpreter(
    #     model_content=model_content)
    interpreter.allocate_tensors()
    input_tensor_details = interpreter.get_input_details()[0]

    if args.input:
        print("Input provided via file...")
        input_tensor = np.fromfile(args.input,
                                   dtype=input_tensor_details["dtype"])
        input_tensor.shape = input_tensor_details["shape"]
    else:
        print("Creating random input...")
        input_tensor = np.array(np.random.random_sample(
            input_tensor_details["shape"]),
                                dtype=input_tensor_details["dtype"])

    interpreter.set_tensor(input_tensor_details["index"], input_tensor)
    print("Invoking tf interpreter...")
    interpreter.invoke()

    num_of_outputs = len(interpreter.get_output_details())
    tflite_outputs = []
    for i in range(num_of_outputs):
        tflite_outputs.append(
            interpreter.get_tensor(
                interpreter.get_output_details()[i]["index"]))

    print("Invoking xformer to get converted model...")
    xformed_model = get_xformed_model(model_content)

    ie = TFLMInterpreter(model_content=xformed_model)
    ie.set_input_tensor(0, input_tensor)
    print("Invoking XCORE interpreter...")
    ie.invoke()
    xformer_outputs = []
    for i in range(num_of_outputs):
        xformer_outputs.append(ie.get_output_tensor(i))

    print("Comparing outputs...")
    for i in range(num_of_outputs):
        print("Comparing output " + str(i) + "...")
        np.testing.assert_equal(tflite_outputs[i], xformer_outputs[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="provide model tflite file")
    parser.add_argument("--input", help="input file")
    args = parser.parse_args()

    test_inference(args)
