# MobileNet v1 Example applications

## Converting flatbuffer to Source File

The following unix command will generate a C source file that contains the TensorFlow Lite model as a char array

    > python ../../../third_party/tensorflow/tensorflow/lite/python/convert_file_to_c_source.py --input_tflite_file ../../models/ImageNet/debug/mobilenet/models/model_xcore.tflite --output_header_file src/mobilenet_v1.h --output_source_file src/mobilenet_v1.c --array_variable_name mobilenet_v1_model --include_guard MOBILENET_V1_MODEL_H_

Note, the command above will overwrite `mobilenet_v1.c`.  In order to allow the model to be stored in flash or DDR, the file needs to be modified after the script creates it.  Add the following lines directly above the line that sets `mobilenet_v1_model[]`.

    #ifdef USE_SWMEM
    __attribute__((section(".SwMem_data")))
    #elif USE_EXTMEM
    __attribute__((section(".ExtMem_data")))
    #endif


## xCORE

Building for xCORE with the model in SRAM

    > make

Building for xCORE with the model in flash

    > make flash

Building for xCORE with the model in DDR

    > make ddr

Running with hardware

    > xrun --io --xscope --xscope-port localhost:10234 bin/mobilenet.xe

Running with simulator

    > xsim --xscope "-realtime localhost:10234" bin/mobilenet.xe

Sending a test image to the xcore.ai Explorer board

    > ./test_image.py path/to/image

## x86

Building for x86

    > make TARGET=x86

Note, `xcore` is the default target.

Running

    > ./bin/mobilenet path/to/image
