Precision Profiling
======================

Summary
-------
This tool can be used to split up a keras model trained with imagenet weights, and evaluate the accuracy of each layer when compared to a tflite and xcore conversion of the model.

Installation
------------

### TFLM Interpreter
For the model you decide to use, operators may need to be added to the resolver in ../../third_party/lib_tflite_micro/tflm_interpreters for the xcore model to run.

You will also need to build the tflm_interpreter, from ../../third_party/lib_tflite_micro/tflm_interpreter run:

  `make install`
### Flatbuffer tflite library
You will also need to build the flatc compiler from the flatbuffers module, and use it to compile the tflite library.

From ../../third_party/lib_tflite_micro/lib_tflite_micro/submodules/flatbuffers run:

  `mkdir build`
  
  `cd build`
  
  `cmake ..`
  
  `make`
  
This will have build flatc. Next to compile the schema run:

  `./flatc --python -o ../../../../../../tools/precision\ profiling/ ../../../../../../tools/precision\ profiling/schema.fbs`
### Virtual Environment and other dependencies  
A virtual environment is reccomended such aw venv. Once an environment is active use requirements.txt to install the required dependencies.

  `pip3 install -r requirements.txt`

  `pip3 install -i https://test.pypi.org/simple/ xmos-tools`

Running model_splitting.py
---------------------------

Due to the high memory requirements of this program, the tool can be run in 3 sections.

First, pass the "generate" option when running model_splitting.py to generate the datasets and models required.

`python3 model_splitting.py generate`

The "evaluate" option will load these models, and run the datasets through the models to produce outputs.

`python3 model_splitting.py evaluate`

The "compare" option will then take these outputs, and compare them to produce various error metrics, and graphs displaying these metrics.

`python3 model_splitting.py compare`
