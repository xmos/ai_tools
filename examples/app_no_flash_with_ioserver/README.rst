Example without flash and using ioserver to communicate with the model from host
================================================================================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile this example, follow these steps::

  python build_model.py
  cmake -G "Unix Makefiles" -B build
  xmake -C build

The first step optimised the ``vww_quant.tflite`` model for xcore; it
produces three files::

  model.tflite
  model.tflite.cpp
  model.tflite.h

The first file is the optimised model; the second file is the generated
source code, the third file is the header for the source code.
The second and third steps configure and build the project.


In order to run this example, follow these steps::

  xrun --xscope bin/app_no_flash_with_ioserver.xe

This runs the app and sets an ioserver via USB that can be communicated to 
from the host.

On Windows, the ``xAISRV`` USB device must use a libusb-compatible driver
before ``run_model.py`` can communicate with it. With the xcore application
running, check that Windows can see the device from PowerShell::

  Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_20B1' } | Format-List

If the ``xAISRV`` device reports that its driver is not installed,
use Zadig to install WinUSB for ``xAISRV``.
Take care to select ``xAISRV`` and not the xTAG debug adapter.

Then run::

  python -m pip install "numpy<2.0" "opencv-python<4.12"

Regardless of the OS, run::

  python run_model.py

``run_model.py`` runs the model via the xcore host interpreter on the host, 
and also on the device using the ioserver via USB.

This script should print::

  Human (98%)
  Not human (1%)
  Connected to XCORE_IO_SERVER via USB
  Human (98%)
  Not human (1%)
