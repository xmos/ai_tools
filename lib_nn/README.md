# Neural Network Kernel Library

## About

We have implemented a library (lib_nn) of efficient neural network functions developed to maximize the performance and minimize the memory footprint of neural network inference on XMOS xcore.ai.

## Required tools and libraries

* xTIMEcomposer Tools Version 15.0.0 or later

## Required hardware

Only XS3 based microcontrollers are supported with this library. The previous generation XS1 and XS2 based microcontrollers are not supported.

XS3 based microcontrollers, like xcore.ai, have a vector unit with 256 bit wide registers and can operate in 8bit, 16bit or 32bit integer mode.

## Prerequisites

This document assumes familiarity with the XMOS xCORE architecture, the XMOS tool chain, the 'C' programming language, and neural network concepts.

## API

The table below gives a quick overview of the APIs in lib_nn.
Unless otherwise noted, all kernels below operate on signed 8-bit input and output tensors.
The following symbols are used:
- Cin - The number of input channels
- Cout - The number of output channels
- Kh - The kernel, filter or pool height
- Kw - The kernel, filter or pool width
- Sh - The stride height
- Sw - The stride width

For full documentation of each API function, please refer to the description in the lib_nn/api/nn_operator.h header file.

| Group | API | VPU Optimized | Constraints | Comments |
|:----|:---|:------------|:----------------|:--------------------------------------------------------|
|Convolution| | | | |
| |conv2d_deep|Yes|Cin % 4 = 0, Cout % 4 = 0 | |
| |conv2d_shallowin|Yes|Cin % 4 = 0, Cout % 4 = 0, Cin * Kw = 32 | |
| |conv2d_1x1|Yes|Cin % 4 = 0, Cout % 4 = 0, Kh = Kw = 1, Sh = Sw = 1 | |
| |conv2d_depthwise|Yes|Cin % 4 = 0, Cout % 4 = 0, Cin = Cout | |
|Fully Connected| | | | |
| |fully_connected_8|Yes|Cin % 4 = 0<sup>1</sup>|Output is 8-bit|
| |fully_connected_16|Yes|Cin % 4 = 0<sup>1</sup>|Output is 16-bit|
|Pooling| | | | |
| |maxpool2d|Yes|Cin % 4 = 0| |
| |avgpool2d|Yes|Cin % 4 = 0| |
| |avgpool2d_global|Yes|Cin % 4 = 0| |
|Argmax| | | | |
| |argmax_16|No|Input is rank-1|Input is 16-bit|
|Activations| | | | |
| |lookup8|No|None|Logistic (sigmoid), tanh & ReLU activation functions can be implemented using a look-up table mapping 8-bit inputs to 8-bit outputs|
|Misc| | | | |
| |add_elementwise|Yes|None| |
| |requantize_16_to_8|Yes|None|Reduces the bit depth of a vector with 16-bit elements to a vector of 8-bit elements|

<sup>1</sup>It is possible to relax this constraint.  See the documentation for the API function.
