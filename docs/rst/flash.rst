Using flash memory
==================

In order to use flash memory with AI tools, your main program needs to be
slightly more complex. We discuss a sequence of options, starting with the
simplest one comprising a single network on a single physical core, and the
final one being a multi-core, multi-network one.

Introduction
------------

XMOS devices enable you to use any type of flash device, as the device
drivers are programmed in software. AI-tools is shipped with a
high-performance flash driver for Quad-flash (also known as QPI or QSPI).
This fast driver measures, at start-up time, the latencies of the
particular flash device and particular XMOS device (due to process
variations and the input voltages on the particular board), and picks a
safe timing window to run the device at high speed; up to 133 MHz, 66
MByte/s.


For each application program
----------------------------

#. Create a directory ``my_models/app_model_flash_single_core`` inside the working directory
 
#. Copy your ``source_model.tflite`` file to this directory
 
#. Run the graph-transformer::
 
     xcore-opt source_model.tflite -o model.tflite --xcore-flash-image-file model.params
 
   This creates four files:
   
   * ``model.tflite`` - optimized tflite model file
 
   * ``model.tflite.cpp`` - optimized C++ model file
 
   * ``model.tflite.h`` - C++ header file which provides API
 
   * ``model.params`` - A file containing all parameters for this model
 
   The parameter file has to be made into a flash image (a file that can be
   written to flash), for this use the ``generate_flash()`` function via the
   Python interface.
 
   .. code-block:: Python
 
     from xmos_ai_tools import xformer as xf
     xf.generate_flash(
         output_file="xcore_flash_binary.out",
         model_files=["model.tflite"],
         param_files=["model.params"]
     )
 
   The flash image .out file can be flashed on XCORE.AI using ``xflash``::
 
     xflash --data xcore_flash_binary.out --target XCORE-AI-EXPLORER
 
   Replace the target with the board that yyou use.
 
#. Next create a src directory with the following main.xc file inside it:
 
   .. literalinclude:: ../../examples/app_flash_single_model/src/main.xc
 
#. Move the output files ``model.tflite.cpp`` and ``model.tflite.h`` into
   the src directory.
 
#. Create a Makefile with the following lines:
 
   .. literalinclude:: ../../examples/app_flash_single_model/Makefile
 
#. Create a file ``config.xscope`` with the following lines:
 
   .. literalinclude:: ../../examples/app_flash_single_model/src/config.xscope
 
#. Source the tools according to your platform (Windows: double click the
   tools icon; Mac: ``source /Applications/XMOS_XTC_<PATH>/Setenv`` Linux:
   ``pushd <PATH-TO-TOOLS>; source SetEnv; popd``)
 
#. Run::
 
     ``xmake``
 
#. Plug an explorer board into your computer and run::
 
     ``xrun --xscope bin/app.xe``
 
#. This should run the network

Programs with more than one model
---------------------------------

If you have multiple Neural Networks, each of which wants to use flash,
then the first thing to do is to transform each in turn::

    xcore-opt source_model1.tflite -o model1.tflite --xcore-flash-image-file model1.params
    xcore-opt source_model2.tflite -o model2.tflite --xcore-flash-image-file model2.params

This creates eight files

   * ``model1.tflite`` - optimized tflite model file for model 1
   * ``model1.tflite.cpp`` - optimized C++ model file for model 1
   * ``model1.tflite.h`` - C++ header file which provides API for model 1
   * ``model1.params`` - A file containing all parameters for model 1
   * ``model2.tflite`` - optimized tflite model file for model 2
   * ``model2.tflite.cpp`` - optimized C++ model file for model 2
   * ``model2.tflite.h`` - C++ header file which provides API for model 2
   * ``model2.params`` - A file containing all parameters for model 2

The parameter files have to be assembled together into a single flash image
(the file that can be written to flash), for this use the
``generate_flash()`` function via the Python interface.

  .. code-block:: Python

    from xmos_ai_tools import xformer as xf
    xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["1.params", "2.params"]
    )

We need a slightly different main.xc:

  .. literalinclude:: ../../examples/app_flash_two_models/src/main.cpp

* Move the output files ``model1.tflite.cpp``, ``model1.tflite.h``,
  ``model2.tflite.cpp`` and ``model2.tflite.h`` into the src directory.

Build and execute the binary

Running two models on two separate cores
----------------------------------------

You need to translate the networks as before, but you need to provide a
multi-core main file that instructs the tool-chain to create different parts
of the program on each core. This multi-core main file is written in XC as
follows:

  .. literalinclude:: ../../examples/app_flash_4/src/main.xc

If you have multiple Neural Networks, each of which wants to use flash,
then the first thing to do is to transform each in turn::

    xcore-opt source_model1.tflite -o model1.tflite --xcore-flash-image-file model1.params
    xcore-opt source_model2.tflite -o model2.tflite --xcore-flash-image-file model2.params

This creates eight files

   * ``model1.tflite`` - optimized tflite model file for model 1
   * ``model1.tflite.cpp`` - optimized C++ model file for model 1
   * ``model1.tflite.h`` - C++ header file which provides API for model 1
   * ``model1.params`` - A file containing all parameters for model 1
   * ``model2.tflite`` - optimized tflite model file for model 2
   * ``model2.tflite.cpp`` - optimized C++ model file for model 2
   * ``model2.tflite.h`` - C++ header file which provides API for model 2
   * ``model2.params`` - A file containing all parameters for model 2

The parameter files have to be assembled together into a single flash image
(the file that can be written to flash), for this use the
``generate_flash()`` function via the Python interface.

  .. code-block:: Python

    from xmos_ai_tools import xformer as xf
    xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["1.params", "2.params"]
    )

* Move the output files ``model1.tflite.cpp``, ``model1.tflite.h``,
  ``model2.tflite.cpp`` and ``model2.tflite.h`` into the src directory.

Build and execute the binary
