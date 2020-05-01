# MobileNet v1 Example applications

## Converting flatbuffer to Source File

The following unix command will generate a C source file that contains the TensorFlow Lite model as a char array

    > python ../../../third_party/tensorflow/tensorflow/lite/python/convert_file_to_c_source.py --input_tflite_file ../../models/ImageNet/debug/mobilenet/models/model_xcore.tflite --output_header_file src/mobilenet_v1.h --output_source_file src/mobilenet_v1.c --array_variable_name mobilenet_v1_model --include_guard MOBILENET_V1_MODEL_H_

## xCORE

Building for xCORE

    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/mobilenet.xe path/to/input.bin path/to/output.bin

## x86

Building for x86

    > make TARGET=x86

Running

    > ./bin/mobilenet path/to/input.bin path/to/output.bin
