# Model Testing Example application

This test application will run inference on a user specified TensorFlow Lite for Microcontrollers
model file (.tflite) using the user specified input.

## xCORE

Building for xCORE

    > mkdir build
    > cd build
    > cmake ../
    > make

Running with simulator

    > xsim --xscope "-realtime localhost:10234" bin/test_model.xe

Running on hardware

    > xrun --io --xscope --xscope-port localhost:10234 bin/test_model.xe

## x86

Building for x86

    > mkdir build
    > cd build
    > cmake ../ -DXCORE=OFF
    > make

Running

    > ./bin/test_model path/to/some.tflite path/to/input path/to/output
