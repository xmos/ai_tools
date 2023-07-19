Example without flash
=====================

If you haven't done so already, download the XMOS AI tools::

  pip3 install xmos_ai_tools

The two flags that maybe useful after ``install`` are ``--upgrade`` (to
upgrade to the latest tools) and ``--dev`` (to upgrade to the latest
development version).

Now initialise your command line environment as follows. For windows::

  FOR /F "delims=" %i IN ('python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))"') DO set XMOS_AITOOLSLIB_PATH=%i

For linux/MacOS::

  export XMOS_AITOOLSLIB_PATH=$(python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))")

And, if you don't have the XMOS tools already, download and install them
from XXX

[All of this is probably to be moved]

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

The third step builds the project

The final step runs the code.


