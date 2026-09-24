Example with two models and learned parameters in flash
=======================================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

This example adds a second model. With a second model, we need to make sure
that we give each model a separate prefix, and we need to merge the two
sets of learned parameters into a single flash image.

In order to compile and run this example follow these steps::

  python export.py
  # For XS3 (XCORE.AI)
  cmake -G "Unix Makefiles" -B build
  # For VX4 (XCORE-400), use this configure command instead
  cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=XK-EVK-XU416
  xmake -C build
  xflash --target XK-EVK-XU316 --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_two_models.xe

This should print::

  No human (9%)
  Human (98%)
