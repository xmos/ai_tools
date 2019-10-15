# CIFAR-10 Example application

## Building

    > xwaf configure clean build

## Running

To run in the simulator

    > xsim --args bin/800MHz/single-op_800MHz.xe mode path/toinput path/to/output

Supported modes are:

    conv2d_deepin_deepout
    fc_deepin_shallowout
