# CIFAR-10 Example applications

## Generating Test Input Images

    > cd test_inputs
    > ./make_test_tensors.py

## Converting flatbuffer to Source File

The following unix command will generate a C source file that contains the TensorFlow Lite model as a char array

    > python ../../../third_party/tensorflow/tensorflow/lite/python/convert_file_to_c_source.py --input_tflite_file ../../models/CIFAR-10/debug/arm_benchmark/models/model_xcore.tflite --output_header_file src/cifar10_model.h --output_source_file src/cifar10_model.c --array_variable_name cifar10_model --include_guard CIFAR10_MODEL_H_

Note, the command above will overwrite `cifar10_model.c`.  In order to allow the model to be stored in flash or DDR, the file needs to be modified after the script creates it.  Add the following lines directly above the line that sets `cifar10_model[]`.

    #ifdef USE_SWMEM
    __attribute__((section(".SwMem_data")))
    #elif USE_EXTMEM
    __attribute__((section(".ExtMem_data")))
    #endif

## xCORE

Building for xCORE with the model in SRAM

    > make TARGET=xcore

Note, `xcore` is the default target.

Building for xCORE with the model in flash

    > make TARGET=xcore flash

Building for xCORE with the model in DDR

    > make TARGET=xcore ddr

Running with the xCORE simulator

    > xsim --xscope "-offline trace.xmt" --args bin/cifar10.xe test_inputs/horse.bin

Running with hardware

    > xrun --io --xscope --args bin/cifar10.xe test_inputs/horse.bin

## x86

Building for x86

    > make TARGET=x86

Running

    > ./bin/arm_benchmark ../test_inputs/ship.bin

## Computing Accuracy on xCORE

    > cd test_inputs
    > ./test_accuracy.py --xe path/to/arm_benchmark.xe
