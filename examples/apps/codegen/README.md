# Code Generation Example applications

## xCORE

**This is currently broken**

Building for xCORE

    > cd codegen
    > make TARGET=xcore

Note, `xcore` is the default target.

Running with the xCORE simulator

    > xsim --args bin/cifar-10.xe path/to/dog.bin

## x86

Building for x86

    > cd codegen
    > make TARGET=x86

Running

    > ./bin/test_model path/to/ship.bin
