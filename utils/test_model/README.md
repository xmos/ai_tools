# Model Testing Example application

This test application will run inference on a user specified TensorFlow Lite for Microcontrollers
model file (.tflite) using the user specified input.

## xCORE

Building all targets for xCORE

    > mkdir build
    > cd build
    > cmake ../
    > make

### Command-line Target

Running with command-line target on simulator

    > xsim --xscope "-offline trace.xmt" --args bin/test_model_cmdline.xe path/to/some.tflite path/to/input path/to/output

Running with command-line target on hardware

    > xrun --io --xscope --args  bin/test_model_cmdline.xe path/to/some.tflite path/to/input path/to/output

### xSCOPE Target

Running with xscope target on simulator

    > xsim --xscope "-realtime localhost:10234" bin/test_model_xscope.xe

Running with xscope target on hardware

    > xrun --io --xscope --xscope-port localhost:10234 bin/test_model_xscope.xe

## x86

Building x86 host target

    > mkdir build
    > cd build
    > cmake ../ -DX86=ON
    > make

### Command-line Target

Running

    > ./bin/test_model_host path/to/some.tflite path/to/input path/to/output
