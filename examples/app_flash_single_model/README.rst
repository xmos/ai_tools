Example with flash
==================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  xcore-opt --xcore-weights-file=model.params vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  cmake -G "Unix Makefiles" -B build
  xmake -C build
  python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model.tflite"],
        param_files=["model.params"]
  )'
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_single_model.xe

When run, the program should print something similar to::

  No human (9%)
  Human (98%)

The difference with the version in ``../app_no_flash`` is that the learned
parameters have been placed into flash memory.
Doing so has significantly reduced the size of the model.
We can see this by looking at the size of the files::

  % ls -l model.*
  -rw-r--r--  1 henk  staff  224576 18 Jul 11:07 model.params
  -rw-r--r--  1 henk  staff   20032 18 Jul 11:07 model.tflite

The python command makes the model.params file into a flash image.
Finally, before running the program, the ``xflash`` command places the
learned parameters into Flash memory on the XCORE-AI-EXPLORER board.
