Example with flash
==================

Please consult XXX on how to install the tools


In order to compile and run this example follow these steps::

  xcore-opt --xcore-flash-image-file=model.params vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  xmake
  python -c 'from xmos_ai_tools import xformer as xf
  xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model.tflite"],
        param_files=["model.params"]
  )'
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_no_flash.xe

This should again print::

  No human (9%)
  Human (98%)

The difference with the version in ``../app_no_flash`` is that we have sent
the learned parameters into flash memory; this has significantly reduced
the size of the model. We can see this by looking at the size of the files::

  % ls -l model.*
  -rw-r--r--  1 henk  staff  224576 18 Jul 11:07 model.params
  -rw-r--r--  1 henk  staff   20032 18 Jul 11:07 model.tflite

The model.params file needs to be made into a flash image, which is what
the python command does. Finally, before we execute it, we must program the
flash with the learned parameters, which is what ``xflash`` is for.


