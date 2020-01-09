# CIFAR-10 Example applications

## Generating Test Input Images

    > cd test_inputs
    > ./make_test_tensors.py

## TensorFLow Lite for Microcontrollers

### xCORE

Building for xCORE

    > cd tflite
    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/cifar-10.xe ../test_inputs/dog.bin

### x86

Building for x86

    > cd tflite
    > make TARGET=x86

Running

    > ./bin/test_model ../test_inputs/ship.bin

## Code Generation

### xCORE

Building for xCORE

    > cd xcore
    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/cifar-10.xe ../test_inputs/dog.bin

### x86

Building for x86

    > cd xcore
    > make TARGET=x86

Running

    > ./bin/test_model ../test_inputs/ship.bin

## Computing Accuracy on xCORE

    > cd test_inputs
    > ./test_accuracy.py --xe path/to/some.xe
