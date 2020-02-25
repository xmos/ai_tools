# Neural Network Kernel Library

## About

We have implemented a library (lib_nn) of efficient neural network functions developed to maximize the performance and minimize the memory footprint of neural network inference on XMOS xcore.ai.

## Required tools and libraries

* xTIMEcomposer Tools Version 15.0.0 or later

## Required hardware

Only XS3 based microcontrollers are supported with this library. The previous generation XS1 and XS2 based microcontrollers are not supported.

XS3 based microcontrollers, like xcore.ai, have a vector unit with 256 bit wide registers and can operates in 8bit, 16bit or 32bit integer mode.

The vector unit implements a hardware ring buffer that allows highly efficient computation of very deep convolutions. In each instruction the vector unit:
* loads a vector of data;
* performs element-wise operations (similarly to SIMD);
* updates a set of result registers.

Instruction results are vectors that can be stored into memory or reduced to scalars. These features make xcore.ai an efficient platform for executing neural network inference.

## Prerequisites

This document assumes familiarity with the XMOS xCORE architecture, the XMOS tool chain, the 'C' programming language, and neural network concepts.

## API

The table below gives a quick overview of the API's in lib_nn. For full documentation of each API function, please refer to the description in the lib_nn/api/nn_operator.h header file.

| Group | API | Constraints | Additional memory required for optimizations (bytes) | VPU Optimized | Comments |
|:----|:---|:------------|:----------------|:--------------------------------------------------------|:-------------|
|Convolution| | | | | |
| |conv2d_deepin_deepout|Input must have a multiple of 32 input channels, and output must have a multiple of 16 output channels|TODO|Yes|Loosening the input and output constraints to multiples of 4 is **in development**|
| |conv2d_shallowin_deepout|Input must have a multiple of 4 input channels, and output must have a multiple of 16 output channels|TODO|Yes|Loosening the input and output constraints to multiples of 4 is **in development**|
| |conv2d_1x1| |TODO|Yes| |
| |depthwise conv2d| | |**In development**| |
|Fully Connected| | | | | |
| |fully_connected_16|Input channels must be a multiple of 4|TODO|Yes| |
|Pooling| | | | | |
| |maxpool2d| |0|Yes| |
| |avgpool2d|Input and output channels must be a multiple of 4|0|Yes| |
| |avgpool2d_global|Input channels must be a multiple of 4|0|Yes| |
|Argmax| | | | | |
| |argmax_16|None|0|No| |
|Softmax| | | | | |
| | | | |**In development**| |
|Activations| | | | | |
| |lookup8|None|256|Yes|Logistic (sigmoid), tanh & ReLU activation functions can be implemented using a look-up table mapping 8-bit inputs to 8-bit outputs|
|Concat| | | | | |
| | | | |**In development**| |
|Misc| | | | | |
| |requantize_16_to_8|Input and output tensors must be word-aligned|0|Yes|Reduces the bit depth of a 16-bit vector to 8 bits|

## Weight & Bias Tensor Layout

TODO: Write something summary-like here!