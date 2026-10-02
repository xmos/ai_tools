Example with two models sharing scratch memory
==============================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

This is an example with two networks, but these two share a scratch memory.

The differences with ``app_flash_two_models`` example are minimal:

* The shared-arena define ``-DSHARED_TENSOR_ARENA`` has been added to the
  CMake build;

* In main.cpp a shared tensor arena is declared::

    uint8_t tensor_arena[LARGEST_TENSOR_ARENA_SIZE] ALIGN(8);

* In main.cpp we ensured that each model is initialised before
  it is invoked; because the arena is shared, each model initialisation
  will overwrite the previous model's data.
  
In order to compile and run this example follow these steps::

  python export.py
  # For XS3 (XCORE.AI)
  cmake -G "Unix Makefiles" -B build
  # For VX4 (XCORE-400), use this configure command instead
  cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=XK-EVK-XU416
  xmake -C build
  xflash --target XK-EVK-XU316 --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_two_models_one_arena.xe

This should print::

  No human (9%)
  Human (98%)



