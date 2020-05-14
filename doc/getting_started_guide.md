# Overview

The XMOS AI tools are comprised of three components:

- tflite2xcore model conversion utility.  
- lib_nn - a library of neural network kernels optimized for the xcore.ai microcontroller.
- A port of the TensorFlow Lite Micro runtime to the xcore.ai microcontroller.

This document explains how to get started with the tflite2xcore model conversion utility.  This utility is used to transform a a quantized TensorFlow Lite model to a format optimized for xcore.ai.  See the [TensorFlow Lite Getting Started Guide](https://www.tensorflow.org/lite/guide/get_started) for instructions on converting and quantizing your TensorFLow model.

**Later this document will be extended with instructions on deploying your converted model to the xcore.ai development kit**

# Requirements

Before installing, make sure your system meets the follow minimum requirements:

- Linux Fedora 30 (or newer) or Ubuntu 18 (or newer)
- MacOS 10.13 High Sierra (or newer)

**Windows is not currently supported.  However, support for Windows is expected for initial product release**

# Installing & Testing tflite2xcore

(Step 1) Create a Conda environment

A [Conda](https://docs.conda.io/) environment is not necessary, however we do recommend that you create one for installing the tflite2xcore Python module.

    > conda create --prefix xmos_env python=3.6

Activate the environment

    > conda activate xcore_env

You may need to specify the fully-qualified path to your environment.

(Step 2) Install tflite2xcore

    > pip install tflite2xcore-0.1.0.tar.gz 

(Step 3) Test the installation

The `xformer.py` script is used to transform a quantized TensorFlow Lite model to a format optimized for xcore.ai.  Run the following command to test the model conversion utility on the example `model_quant.tflite` test model provided:

    > xformer.py model_quant.tflite model_xcore.tflite

Included in the installation, the `tflite_visualize.py` script can be used to visualize a TensorFlow Lite model, including those converted for xcore.ai.  You can visualize the test model conversion with the following command:

    > tflite_visualize.py model_xcore.tflite -o model_xcore.html

Open `model_xcore.html` in your browser to inspect the model.

# Converting Your Model

Follow step (3) in "Installing & Testing tflite2xcore".  But, substitute the example files provided with fully-qualified paths to your TensorFlow Lite quantized model.

# TensorFlow Lite Operators Optimized for xcore.ai

Below is a list of TensorFlow Lite Operators that have been optimized for xcore.ai.  Depending on the parameters of the operator, the optimized implementation will be 10-50 times faster than the reference implementation, which will be used for all other operators.

- CONV_2D
- FULLY_CONNECTED
- MAX_POOL_2D
- DEPTHWISE_CONV_2D
- AVERAGE_POOL_2D
- MEAN
- LOGISTIC
- RELU
- RELU6

Additional operators will be optimized in future releases.

# Deploying Your Model

Development kits will be available soon. Register at xcore.ai for all the latest news - including early announcement of the first available development kits.