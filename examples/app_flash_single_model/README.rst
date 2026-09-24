Example with flash
==================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  python export.py
  # For XS3 (XCORE.AI)
  cmake -G "Unix Makefiles" -B build
  # For VX4 (XCORE-400), use this configure command instead
  cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=XK-EVK-XU416
  xmake -C build
  xflash --target XK-EVK-XU316 --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_single_model.xe

When run, the program should print something similar to::

  No human (9%)
  Human (98%)

The difference with the version in ``../app_no_flash`` is that the learned
parameters have been placed into flash memory.
Doing so has significantly reduced the size of the model.
We can see this by looking at the size of the files::

  % ls -l model.params src/model.tflite
  -rw-r--r--  1 henk  staff  224576 18 Jul 11:07 model.params
  -rw-r--r--  1 henk  staff   20032 18 Jul 11:07 model.tflite

The export script makes the model.params file into a flash image.
Finally, before running the program, the ``xflash`` command places the
learned parameters into Flash memory on the XK-EVK-XU316 board.
