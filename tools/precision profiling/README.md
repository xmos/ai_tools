Precision Profiling
======================

Summary
-------
This tool can be used to split up a keras model trained with imagenet weights, and evaluate the accuracy of each layer when compared to a tflite and xcore conversion of the model.

Installation
------------

For the model you decide to use, operators may need to be added to the resolver in ../../third_party/lib_tflite_micro/tflm_interpreters for the xcore model to run.

You will also need to build the tflm_interpreter.

A virtual environment is reccomended, once in this environment use requirements.txt to install the required dependencies.

Running model_splitting.py
---------------------------

Due to the high memory requirements of this program, the tool can be run in 3 sections.

First, pass the "generate" option when running model_splitting.py to generate the datasets and models required.

The "evaluate" option will load these models, and run the datasets through the models to produce outputs.

The "compare" option will then take these outputs, and compare them to produce various error metrics, and graphs displaying these metrics.
