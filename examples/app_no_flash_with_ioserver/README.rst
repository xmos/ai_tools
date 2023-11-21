Example without flash and using ioserver to communicate with the model from host
======================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile this example, follow these steps::

  python build_model.py
  xmake

The first step optimised the ``vww_quant.tflite`` model for xcore; it
produces three files::

  model.tflite
  model.tflite.cpp
  model.tflite.h

The first file is the optimised model; the second file is the generated
source code, the third file is the header for the source code.
The second step builds the project.


In order to run this example, follow these steps::

  xrun --xscope bin/app_no_flash_with_ioserver.xe

This runs the app and sets an ioserver via USB that can be communicated to 
from the host.

Then run::
  
  python run_model.py

``run_model.py`` runs the model via the xcore host interpreter on the host, 
and also on the device using the ioserver via USB.

This should print::

  Human (98%)
  Not human (1%)
  Connected to XCORE_IO_SERVER via USB
  Human (98%)
  Not human (1%)
