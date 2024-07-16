Example with flash and tensor arena on external memory
==================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

This is an example with one network using a tensor arena in external memory.

The differences with the ``app_flash_single_model`` example are minimal:

* The shared-arena definition ``-DSHARED_TENSOR_ARENA`` has been added to the
  Makefile; this allows the application to define the tensor arena buffer.

* In support.cpp a tensor arena is declared in external memory::\

    __attribute__ ((section(".ExtMem.bss")))
    uint8_t tensor_arena[LARGEST_TENSOR_ARENA_SIZE] __attribute__((aligned(8)));

In order to compile and run this example follow these steps::

  xcore-opt --xcore-weights-file=model.params vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  xmake
  python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model.tflite"],
        param_files=["model.params"]
  )'
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_single_model.xe

This should print::

  No human (9%)
  Human (98%)
