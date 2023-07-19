Example without flash
=====================

Please consult `here <../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  xcore-opt vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  xmake
  xrun --xscope bin/app_no_flash.xe

This should print::

  No human (9%)
  Human (98%)

The first step optimised the ``vww_quant.tflite`` model  for xcore; it
produces three files::

  model.tflite
  model.tflite.cpp
  model.tflite.h

The first file is the optimised model; the second file is the generated
source code, the third file is the header for the source code.

The second step places the source code into the source directory.

The third step builds the project.

The final step runs the code.


