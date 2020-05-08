# Requirements

TODO

MacOS 10.13 High Sierra (or newer)

# Installing & Testing tflite2xcore

(1) Create a Conda the environment

A Conda environment is not necessary, however we do recommend that you create one for installing the tflite2xcore Python module.

    > conda create --prefix xmos_env python=3.6

Activate the environment

    > conda activate xcore

(2) Install tflite2xcore

    > pip install tflite2xcore-0.1.0.tar.gz 

(3) Test the installation

The `xformer.py` script is used to transform a quantized TensorFlow Lite model to a format optimized for xcore.ai. Run the following command to test the transformer on the `model_quant.tflite` test model provided.
  
    > xformer.py model_quant.tflite model_xcore.tflite

The `tflite_visualize.py` script can be used to visualize a TensorFlow Lite model, including those converted for xcore.ai.  You can visualize the test model conversion with the following command.

    > tflite_visualize.py model_xcore.tflite -o model_xcore.html

# Converting Your Model

Follow step (3) in "Installing & Testing tflite2xcore".  But, substitute the example files provided with fully-qualified paths to your TensorFlow Lite quantized model.

# TensorFlow Lite Operators Optimized for xcore.ai

Below is a list of TensorFlow Lite Operators that have been optimized for xcore.ai.  Depending on the parameters of the operator, the optimized implementation will be 10-50 times faster than the reference implementation.

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

No deployment steps can be taken t this time.