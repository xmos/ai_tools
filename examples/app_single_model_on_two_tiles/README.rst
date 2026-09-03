Example single model on two tiles
=================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  python generate_optimized_cpp_for_xcore.py
  cmake -G "Unix Makefiles" -B build
  xmake -C build
  xrun --xscope bin/app_device.xe

When run, the program should print a considerable number of lines
concluding with something similar to::

  Class with max1 value = 291 and probability = 0.839844
  Class with max2 value = 200 and probability = 0.011719
  Class with max3 value = 160 and probability = 0.007812

The first step optimises the ``mobilenetv1_25.tflite`` model for xcore;
it produces these generated files::

  src/model.tflite
  src/model.tflite.cpp
  src/model.tflite.h
  src/model_weights.c
  src/model_weights.h

The ``model.tflite`` file contains the optimised model,
the ``model.tflite.cpp`` file contains the generated source code,
the ``model.tflite.h`` file contains the header for the source code, and
the ``model_weights.c`` file contains weights that are served from the other tile.
The ``model_weights.h`` file contains constants used by the generated weights source.

The configure and build steps build the project using xcommon-cmake.

The final step runs the code.
