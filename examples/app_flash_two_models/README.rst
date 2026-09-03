Example with two models and learned parameters in flash
=======================================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

This example adds a second model. With a second model, we need to make sure
that we give each model a separate prefix, and we need to merge the two
sets of learned parameters into a single flash image.

In order to compile and run this example follow these steps::

  xcore-opt --xcore-weights-file=model1.params \
            --xcore-naming-prefix=model1_ \
            vww_quant1.tflite -o model1.tflite
  xcore-opt --xcore-weights-file=model2.params \
            --xcore-naming-prefix=model2_ \
            vww_quant2.tflite -o model2.tflite
  mv model1.tflite.cpp model1.tflite.h src
  mv model2.tflite.cpp model2.tflite.h src
  # For XS3 (XCORE.AI)
  cmake -G "Unix Makefiles" -B build
  # For VX4 (XCORE-400), use this configure command instead
  cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=XK-EVK-XU416
  xmake -C build
  python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["model1.params", "model2.params"]
  )'
  xflash --target XK-EVK-XU316 --data xcore_flash_binary.out
  xrun --xscope bin/app_flash_two_models.xe

This should print::

  No human (9%)
  Human (98%)
