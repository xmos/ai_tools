YOLOv8 example with flash
=========================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

Setup
-----

Install the Python dependencies for this example after installing
``xmos-ai-tools``::

  python -m pip install -r requirements.txt

The requirements file includes additional version bounds for the Python export
path used by this example. Run the requirements install after installing
``xmos-ai-tools`` so these bounds override any incompatible transitive packages
selected by the base tools installation.

The model generation script disables Ultralytics package auto-installation to
avoid changing dependencies at runtime. If a previous run upgraded ``protobuf``,
rerun ``python -m pip install -r requirements.txt`` before generating the model.

Build and run
-------------

Generate the model sources, configure the xcommon-cmake build, build the
application, flash the model data, and run the application::

  python obtain_and_optimize_yolov8_cls.py
  cmake -G "Unix Makefiles" -B build
  xmake -C build
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_yolov8_classification.xe

Generated files
---------------

The model generation step downloads or creates these files::

  yolov8n-cls.pt
  yolov8n-cls.onnx
  yolov8n-cls_saved_model/
  src/model.tflite
  src/model.tflite.cpp
  src/model.tflite.h
  src/model_flash.params
  xcore_flash_binary.out

The ``src/model.tflite.cpp`` and ``src/model.tflite.h`` files are generated
source files used by the CMake build. The ``src/model_flash.params`` and
``xcore_flash_binary.out`` files contain the model weights for flash.

After Ultralytics exports ONNX, the script runs a strict full-int8 TensorFlow
Lite conversion with the onnx2tf TensorFlow converter backend before invoking
``xcore-opt``. This avoids passing hybrid TFLite models to TFLite Micro.

This example also includes a compatibility shim for pre-release ``xmos-ai-tools``
wheels that import ``opcode2name`` from older ``tflite`` package layouts.

Output
------

In the example, we inference the model with a sample image of a LION. 
Running the example should print::

  Correct - Inferred class is LION!

The same lion image is saved in raw format as ``lion.bin`` for inference on the
host interpreter, and as a header file in ``src/lion.h`` for inference on device.
