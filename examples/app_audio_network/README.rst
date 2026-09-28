App Audio Network
=================

This example demonstrates an audio denoising network running as part of a USB
audio application. It runs on the XCORE.AI MC Audio Board.

Setup
-----

1. Ensure you have XTC tools version 15.3.1 activated in your current terminal.
2. Install the ``xmos_ai_tools`` Python package in your virtual environment (venv).

Build and Run
-------------

Run the following commands in the current directory.

.. code-block:: console

    # generate model sources
    python export.py

    # build
    # For XS3 (XCORE.AI)
    cmake -G "Unix Makefiles" -B build
    # For VX4 (XCORE-400), use this configure command instead
    cmake -G "Unix Makefiles" -B build -DAPP_HW_TARGET=src/core/xk-audio-416-mc.xn
    xmake -C build

    # run
    xrun --xscope bin/app_audio_network.xe

Generated Files
---------------

The model generation step optimises the ``denoise_16x8.tflite`` model for
xcore and produces these files::

  src/model_audioi16.tflite
  src/model_audioi16.tflite.cpp
  src/model_audioi16.tflite.h

The first file contains the optimised model, the second file contains the
generated source code, and the third file contains the header for the source
code.

The export script places the generated source code and header directly into the
source directory, where they are consumed by the CMake build.

Output
------

The application should enumerate as a USB audio device. Select the XCORE.AI MC
Audio Board as the host audio device to send audio through the denoising path.
