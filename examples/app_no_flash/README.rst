Example without flash
=====================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  xcore-opt vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  cmake -G "Unix Makefiles" -B build
  xmake -C build
  xrun --xscope bin/app_no_flash.xe

When run, the program should print something similar to::

  No human (9%)
  Human (98%)

The first step optimises the ``vww_quant.tflite`` model for xcore;
it produces three files::

  model.tflite
  model.tflite.cpp
  model.tflite.h

The first file contains the optimised model,
the second file contains the generated source code, and
the third file contains the header for the source code.

The second step places the generated source code into the source directory.

The configure and build steps build the project.

The final step runs the code.
