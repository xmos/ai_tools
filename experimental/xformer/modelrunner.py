# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import os
import tempfile
import sys
import subprocess
import numpy as np
import pathlib
import argparse
from cv2 import cv2

import tensorflow
import larq_compute_engine as lce
from tflm_interpreter import TFLMInterpreter

XFORMER2_PATH = (pathlib.Path(__file__).resolve().parents[0] / "bazel-bin" /
                 "xcore-opt")


def get_xformed_model(model):
    with tempfile.TemporaryDirectory(suffix=str(os.getpid())) as dirname:
        input_path = pathlib.Path(dirname) / "input.tflite"

        with open(pathlib.Path(input_path).resolve(), "wb") as fd:
            fd.write(model)

        output_path = pathlib.Path(dirname) / "output.tflite"
        cmd = [str(XFORMER2_PATH), str(input_path), "-o", str(output_path),
        #"--xcore-replace-avgpool-with-conv2d",
        #"--xcore-replace-with-conv2dv2",
        #"--xcore-translate-to-customop"
        ]
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

    # for attaching a debugger
    if args.s:
        import time
        time.sleep(5)

    if args.bnn:
        print("Creating LCE interpreter...")
        interpreter = lce.testing.Interpreter(model_content)
        input_tensor_type = interpreter.input_types[0]
        input_tensor_shape = interpreter.input_shapes[0]

    else:
        print("Creating TFLite interpreter...")
        interpreter = tensorflow.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tensorflow.lite.experimental.
            OpResolverType.BUILTIN_REF, experimental_preserve_all_tensors=True)
        # interpreter = tensorflow.lite.Interpreter(
        #     model_content=model_content)
        interpreter.allocate_tensors()
        input_tensor_details = interpreter.get_input_details()[0]
        input_tensor_type = input_tensor_details["dtype"]
        input_tensor_shape = input_tensor_details["shape"]

    if args.input:
        print("Input provided via file...")
        s = input_tensor_details["shape"]
        img = cv2.imread(args.input)
        res = cv2.resize(img, dsize=(s[1], s[2]), interpolation=cv2.INTER_CUBIC)

        input_tensor = np.array(res, dtype=input_tensor_type)
        input_tensor.shape = input_tensor_shape
    else:
        print("Creating random input...")
        input_tensor = np.array(100 * np.random.random_sample(
            input_tensor_shape), dtype=input_tensor_type)

    #print(repr(input_tensor))

    if args.bnn:
        print("Invoking LCE interpreter...")
        outputs = interpreter.predict(input_tensor)
        # for some reason, batch dim is missing in lce output
        outputs = [outputs]
        num_of_outputs = len(outputs)
        print(outputs)
    else:
        interpreter.set_tensor(input_tensor_details["index"], input_tensor)
        print("Invoking TFLite interpreter...")
        interpreter.invoke()

        num_of_outputs = len(interpreter.get_output_details())
        outputs = []
        for i in range(num_of_outputs):
            outputs.append(
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
        np.testing.assert_equal(outputs[i], xformer_outputs[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="provide model tflite file")
    parser.add_argument("--input", help="input file")
    parser.add_argument("--bnn", default=False, action='store_true', help="run bnn")
    parser.add_argument("--s", default=False, action='store_true', help="sleep for 5 seconds")
    args = parser.parse_args()
    test_inference(args)
