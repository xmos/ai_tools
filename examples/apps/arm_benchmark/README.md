# ARM Benchmark Example applications

## Generating Test Input Images

    > cd test_inputs
    > ./make_test_tensors.py

## Converting flatbuffer to Source File

The following unix command will generate a C source file that contains the TensorFlow Lite model as a char array

    > python ../../../third_party/tensorflow/tensorflow/lite/python/convert_file_to_c_source.py --input_tflite_file ../../models/CIFAR-10/debug/arm_benchmark/models/model_xcore.tflite --output_header_file src/xcore_model.h --output_source_file src/xcore_model.c --array_variable_name xcore_model --include_guard XCORE_MODEL_H_

## xCORE

Building for xCORE

    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/arm_benchmark.xe ../test_inputs/dog.bin

## x86

Building for x86

    > make TARGET=x86

Running

    > ./bin/arm_benchmark ../test_inputs/ship.bin

## Computing Accuracy on xCORE

    > cd test_inputs
    > ./test_accuracy.py --xe path/to/arm_benchmark.xe
