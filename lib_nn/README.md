# Neural Network Kernel Library

## About

We have implemented a library (lib_nn) of efficient neural network functions developed to maximize the performance and minimize the memory footprint of neural network inference on XMOS xcore.ai.

## Required tools and libraries

* xTIMEcomposer Tools Version 15.0.0 or later

## Required hardware

Only XS3 based microcontrollers are supported with this library. The previous generation XS1 and XS2 based microcontrollers are not supported.

XS3 based microcontrollers, like xcore.ai, have a vector unit with 256 bit wide registers and can operate in 8bit, 16bit or 32bit integer mode.

The vector unit implements a hardware ring buffer that allows highly efficient computation of very deep convolutions. In each instruction the vector unit:
* loads a vector of data;
* performs element-wise operations (similarly to SIMD);
* updates a set of result registers.

Instruction results are vectors that can be stored into memory or reduced to scalars. These features make xcore.ai an efficient platform for executing neural network inference.

## Prerequisites

This document assumes familiarity with the XMOS xCORE architecture, the XMOS tool chain, the 'C' programming language, and neural network concepts.

## API

The table below gives a quick overview of the APIs in lib_nn.
Unless otherwise noted, all kernels below operate on signed 8-bit input and output tensors.
The following symbols are used:
- Cin - The number of input channels
- Cout - The number of output channels
- Kh - Kernel/filter/pool height
- Kw - Kernel/filter/pool width

For full documentation of each API function, please refer to the description in the lib_nn/api/nn_operator.h header file.

| Group | API | VPU Optimized | Constraints    | Additional memory required for optimization (bytes) | Comments |
|:----|:---|:------------|:----------------|:--------------------------------------------------------|:-------------|
|Convolution| | | | | |
| |conv2d_deepin_deepout|**In re-development**|Cin % 4 = 0, Cout % 4 = 0|(160 * (Cout + 15) >> 4) - (Cout * 4)| |
| |conv2d_shallowin_deepout|**In re-development**|Cin % 4 = 0, Cout % 4 = 0, Cin * Kh = 32|(160 * (Cout + 15) >> 4) - (Cout * 4)| |
| |conv2d_1x1|Yes|Cin % 4 = 0, Cout % 4 = 0, Kh = Kw = 1|(160 * (Cout + 15) >> 4) - (Cout * 4)|
| |depthwise conv2d|**In development**|Cin % 4 = 0, Cout % 4 = 0, Cin = Cout|(160 * (Cout + 15) >> 4) - (Cout * 4)| |
|Fully Connected| | | | | |
| |fully_connected_16|Yes|Cin % 4 = 0, Cout % 4 = 0|(160 * (Cout + 15) >> 4) - (Cout * 4)|Output is 16-bit|
|Pooling| | | | | |
| |maxpool2d|Yes|None|0| |
| |avgpool2d|Yes|Cin % 4 = 0, Cout % 4 = 0|0| |
| |avgpool2d_global|Yes|Cin % 4 = 0, Cout % 4 = 0|0| |
|Argmax| | | | | |
| |argmax_16|No|Input is rank-1|0|Input is 16-bit|
|Activations| | | | | |
| |lookup8|No|None|256|Logistic (sigmoid), tanh & ReLU activation functions can be implemented using a look-up table mapping 8-bit inputs to 8-bit outputs|
|Misc| | | | | |
| |requantize_16_to_8|Yes|Cin % 4 = 0, Cout % 4 = 0|0|Reduces the bit depth of a 16-bit vector to 8 bits|

