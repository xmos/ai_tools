MobilenetV2 example with flash
==================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  python obtain_and_build_mobilenetv2.py
  xmake
  xflash --target XCORE-AI-EXPLORER --data xcore_flash_binary.out
  xrun --xscope bin/app_mobilenetv2.xe

In the example, we inference the model with a sample image of a LION. 
Running the example should print::

  Correct - Inferred class is LION!

The same lion image is saved in raw format for inference on the host interpreter, and as a header file in ``src/lion.h`` for inference on device.
