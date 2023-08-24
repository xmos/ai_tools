# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import os
import tempfile
import sys
import subprocess
import numpy as np
import pathlib
import argparse
import cv2
from itertools import chain

import tensorflow as tf
import larq_compute_engine as lce
from xmos_ai_tools.xinterpreters import (
    xcore_tflm_host_interpreter,
    xcore_tflm_usb_interpreter,
)


def checksum_calc(data):
    res = np.uint8(0)
    for i in range(0, len(data)):
        res -= np.uint8(data[i])
    return res


XFORMER2_PATH = pathlib.Path(__file__).resolve().parents[0] / "bazel-bin" / "xcore-opt"


def dequantize(arr: np.ndarray, scale: float, zero_point: int) -> np.float32:
    return np.float32(arr.astype(np.int32) - np.int32(zero_point)) * np.float32(scale)


def get_xformed_model(model, args):
    with tempfile.TemporaryDirectory(suffix=str(os.getpid())) as dirname:
        input_path = pathlib.Path(dirname) / "input.tflite"

        with open(pathlib.Path(input_path).resolve(), "wb") as fd:
            fd.write(model)

        params_path = pathlib.Path(dirname) / "output.params"
        output_path = pathlib.Path(dirname) / "output.tflite"
        cmd = [
            str(XFORMER2_PATH),
            str(input_path),
            "-o",
            str(output_path),
            "--xcore-thread-count=" + args.tc,
            "--xcore-flash-image-file=" + str(params_path),
            # "--lce-translate-tfl",
            # "--xcore-replace-with-conv2dv2",
            # "--xcore-translate-to-customop"
            # "--xcore-op-split-tensor-arena",
            # "--xcore-op-split-bottom-op=16,24",
            # "--xcore-op-split-top-op=6,18",
            # "--xcore-op-split-num-splits=8,6",
            # "--xcore-conv-err-threshold=3.6",
            # "--xcore-offline-offsets=1",
            # "--xcore-overlap=1"
        ]
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
        )
        print(p.stdout)
        with open(pathlib.Path(output_path).resolve(), "rb") as fd:
            output_bits = bytes(fd.read())
        with open(pathlib.Path(params_path).resolve(), "rb") as fd:
            params_bits = bytes(fd.read())
    return output_bits, params_bits


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
        interpreter = lce.testing.Interpreter(
            model_content, num_threads=1, use_reference_bconv=True
        )
        # interpreter = lce.testing.Interpreter(model_content, num_threads=1)
        input_tensor_type = interpreter.input_types[0]
        input_tensor_shape = interpreter.input_shapes[0]

    else:
        print("Creating TFLite interpreter...")
        # interpreter = tf.lite.Interpreter(
        #     model_content=model_content,
        #     experimental_op_resolver_type=tf.lite.experimental.
        #     OpResolverType.BUILTIN_REF, experimental_preserve_all_tensors=True)
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        num_of_inputs = len(interpreter.get_input_details())
        input_tensor_type = []
        input_tensor_shape = []
        for i in range(num_of_inputs):
            input_tensor_type.append(interpreter.get_input_details()[i]["dtype"])
            input_tensor_shape.append(interpreter.get_input_details()[i]["shape"])

        # input_tensor = np.array(100 * np.random.random_sample(input_tensor_shape), dtype=input_tensor_type)
        # interpreter.set_tensor(input_tensor_details["index"], input_tensor)
        # print("Invoking TFLite interpreter...")
        # interpreter.invoke()

    if args.input:
        print("Input provided via file...")
        s = input_tensor_shape
        img = cv2.imread(args.input)
        res = cv2.resize(img, dsize=(s[1], s[2]), interpolation=cv2.INTER_CUBIC)

        input_tensor = np.array(res, dtype=input_tensor_type)
        input_tensor.shape = input_tensor_shape

    # print(repr(input_tensor))
    print("Invoking xformer to get converted model...")
    xformed_model, params = get_xformed_model(model_content, args)

    print("Creating TFLM XCore interpreter...")
    if args.device:
        ie = xcore_tflm_usb_interpreter()
    else:
        ie = xcore_tflm_host_interpreter()
    ie.set_model(
        model_content=xformed_model, params_content=params, secondary_memory=True
    )
    # ie.set_model(model_content=xformed_model, secondary_memory=True)

    if args.cifar:
        (_, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
        # shift images from 0-255 to -128-127
        test_images = tf.cast(test_images - 128.0, tf.int8)

    # Run tests
    num_of_fails = 0
    for test in range(0, int(args.n)):
        print("Run #" + str(test))
        input_tensor = []
        if args.cifar:
            print("Using cifar images...")
            # add batch dim
            input_tensor = np.expand_dims(test_images[test], axis=0)
        else:
            print("Creating random input...")
            # print(k)
            # k = np.load("1.npy")
            for i in range(num_of_inputs):
                k = []
                n = -128
                for j in range(0, np.prod(input_tensor_shape[i])):
                    if n >= 128:
                        n = -128
                    k.append(n)
                    n = n + 3
                # input_tensor.append(np.array(255 * np.random.random_sample(input_tensor_shape[i]) - 128, dtype=input_tensor_type[i]))
                # input_tensor.append(np.array(1 * np.ones(input_tensor_shape[i]), dtype=input_tensor_type[i]))
                input_tensor.append(
                    np.reshape(
                        np.asarray(k, dtype=input_tensor_type[i]), input_tensor_shape[i]
                    )
                )

        if args.bnn:
            print("Invoking LCE interpreter...")
            outputs = interpreter.predict(input_tensor)
            # for some reason, batch dim is missing in lce when only one output
            if len(outputs) == 1:
                outputs = [outputs]
            num_of_outputs = len(outputs)
        else:
            for i in range(num_of_inputs):
                interpreter.set_tensor(
                    interpreter.get_input_details()[i]["index"], input_tensor[i]
                )
            print("Invoking TFLite interpreter...")
            interpreter.invoke()

            num_of_outputs = len(interpreter.get_output_details())
            outputs = []
            output_scales = []
            output_zero_points = []
            for i in range(num_of_outputs):
                outputs.append(
                    interpreter.get_tensor(interpreter.get_output_details()[i]["index"])
                )
                quant_params = interpreter.get_output_details()[i][
                    "quantization_parameters"
                ]
                output_scales.append(quant_params["scales"])
                output_zero_points.append(quant_params["zero_points"])

        # print("Creating 2nd LCE interpreter...")
        # ie = lce.testing.Interpreter(xformed_model, num_threads=1, use_reference_bconv=True)
        # outputs2 = ie.predict(input_tensor)
        # # for some reason, batch dim is missing in lce when only one output
        # if len(outputs2) == 1:
        #     outputs2 = [outputs2]
        # print(outputs2)

        print("Invoking XCORE interpreter...")
        for i in range(num_of_inputs):
            ie.set_tensor(i, input_tensor[i])
        ie.invoke()
        xformer_outputs = []
        for i in range(num_of_outputs):
            xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]["index"]))

        # Compare outputs
        for i in range(num_of_outputs):
            print("Comparing output number " + str(i) + "...")
            try:
                print("xformer output")
                print(xformer_outputs[i])
                print("checksum")
                print(checksum_calc(xformer_outputs[i].flatten()))
                print("compared output")
                print(outputs[i])
                print("checksum")
                print(checksum_calc(outputs[i].flatten()))

                errors = np.array(
                    list(
                        chain(
                            *[
                                np.array(a - b).reshape(-1)
                                for a, b in zip(outputs, xformer_outputs)
                            ]
                        )
                    )
                ).reshape(-1)
                print(np.sum(errors))
                # if quantized output, we dequantize it before comparing
                if output_scales[i]:
                    outputs[i] = dequantize(
                        outputs[i], output_scales[i], output_zero_points[i]
                    )
                    xformer_outputs[i] = dequantize(
                        xformer_outputs[i], output_scales[i], output_zero_points[i]
                    )
                np.testing.assert_equal(outputs[i], xformer_outputs[i])
            except Exception as e:
                num_of_fails += 1
                print(e)
                print("Run #" + str(test) + " failed")
            # np.testing.assert_equal(outputs[i], outputs2[i])
    if args.device:
        ie.close()
    return num_of_fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="provide model tflite file")
    parser.add_argument("--input", help="input file")
    parser.add_argument("--bnn", default=False, action="store_true", help="run bnn")
    parser.add_argument(
        "--device", default=False, action="store_true", help="run on xcore"
    )
    parser.add_argument(
        "--s", default=False, action="store_true", help="sleep for 5 seconds"
    )
    parser.add_argument(
        "--cifar", default=False, action="store_true", help="enable cifar test data"
    )
    parser.add_argument("--n", default=1, help="num of runs")
    parser.add_argument("--tc", help="thread count")
    args = parser.parse_args()

    num_of_fails = test_inference(args)
    print("\nTotal tests = " + str(args.n))
    print("Total fails = " + str(num_of_fails))
