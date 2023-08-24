Example with two models sharing scratch memory
==============================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

This is an example with two networks, but these two share a scratch memory.

The differences with ``app_flash_two_models`` example are minimal:

* The shared-arena defined ``-DSHARED_TENSOR_ARENA`` has been added to the
  Makefile;

* In main.cpp a shared tensor arena is declared::

    uint8_t tensor_arena[LARGEST_TENSOR_ARENA_SIZE] ALIGN(8);

* In main.cpp we ensured that each model is initialised before
  it is invoked; because the arena is shared, each model initialisation
  will overwrite the previous model's data.
  
In order to compile and run this example follow these steps::

  xcore-opt --xcore-flash-image-file=model1.params \
            --xcore-naming-prefix=model1_ \
            vww_quant1.tflite -o model1.tflite
  xcore-opt --xcore-flash-image-file=model2.params \
            --xcore-naming-prefix=model2_ \
            vww_quant2.tflite -o model2.tflite
  mv model1.tflite.cpp model1.tflite.h src
  mv model2.tflite.cpp model2.tflite.h src
  xmake
  python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["model1.params", "model2.params"]
  )'
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_two_models_one_arena.xe

This should print::

  No human (9%)
  Human (98%)



