Example with two models and learned parameters flash
====================================================

Please consult XXX on how to install the tools

This example adds a second model. With a second model, we need to make sure
that we give each model a separate prefix, and we need to merge the two
sets of learned parameters into a single flash image.

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
  python -c 'from xmos_ai_tools import xformer as xf
  xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["model1.params", "model2.params"]
  )'
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_no_flash.xe

This should again print::

  No human (9%)
  Human (98%)


