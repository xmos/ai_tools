Work Flow
=========

``xmos-ai-tools`` is available on https://pypi.org/project/xmos-ai-tools/.
It includes:

* the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
* the XCore tflm interpreter to run the transformed models on host

Initial steps
-------------

Perform the following steps once

* Install xmos-ai-tools::

    # Create a virtual environment with
    python3 -m venv <name_of_virtualenv>

    # Activate the virtual environment
    # On Windows, run:
    <name_of_virtualenv>\Scripts\activate.bat
    # On Linux and MacOS, run:
    source <name_of_virtualenv>/bin/activate

    # Install xmos-ai-tools from PyPI
    pip3 install xmos-ai-tools --upgrade

  Use ``pip3 install xmos-ai-tools --pre --upgrade`` instead if you want to install the latest development version.

  Installing ``xmos-ai-tools`` will make the xcore-opt binary available in your shell to use directly, or you can use the Python interface as detailed `here <develop/docs/rst/python.rst>`_.

* Obtain the tool-chain from http://www.xmos.ai/tools and install it according to the platform instructions.

* Setup ``XMOS_AITOOLSLIB_PATH`` environment variable. This is used to identify the installed location of xmos-ai-tools library and headers.

  On Windows, run the following command::

    FOR /F "delims=" %i IN ('python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))"') DO set XMOS_AITOOLSLIB_PATH=%i

  On MacOS and Linux, run the following command::

    export XMOS_AITOOLSLIB_PATH=$(python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))")


* Create a sandbox (working directory)


For each application program
----------------------------

* Create a directory my_models/app_model_1 inside the working directory

* Copy your ``source_model.tflite`` file to this directory

* Run the graph-transformer::

    xcore-opt source_model.tflite -o model.tflite

  This creates three files
   * ``model.tflite`` - optimized tflite model file
   * ``model.tflite.cpp`` - optimized C++ model file
   * ``model.tflite.h`` - C++ header file which provides API

  To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-flash-image-file" option.
  ``model1.tflite`` and ``model2.tflite`` are example models to demonstrate the API::

   xcore-opt source_model.tflite -o model1.tflite --xcore-flash-image-file 1.params

   xcore-opt source_model.tflite -o model2.tflite --xcore-flash-image-file 2.params


  To combine these files created above into a flash image .out file, use the generate_flash() function via the Python interface.

  .. code-block:: Python

    from xmos_ai_tools import xformer as xf
    xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["1.params", "2.params"]
    )

  The flash image .out file can be flashed on XCORE.AI using ``xflash``::

    xflash --data xcore_flash_binary.out --target XCORE-AI-EXPLORER


* Create a src directory with the following main.c file inside it::

    #include <model.tflite.h>

    int main(void) {
      model_init(null);
      int8_t *inputs = model_input_ptr(0);
      // copy data to inputs
      model_invoke();
      int8_t *outputs = model_output_ptr(0);
      // print outputs.
    }

* Move the output files ``model.tflite.cpp`` and ``model.tflite.h`` into
  the src directory.

* Create a Makefile with the following lines::

    TARGET=XCORE-AI-EXPLORER
    APP_NAME=app.xe
    XCC_FLAGS=-fxscope
    USED_MODULES = lib_tflite_micro
    XMOS_MAKE_PATH ?= ../..
    include $(XMOS_MAKE_PATH)/xcommon/module_xcommon/build/Makefile.common

* Create a file ``config.xscope`` with the following lines::

    <?xml version="1.0" encoding="UTF-8"?>

    <xSCOPEconfig ioMode="basic" enabled="true"/>

* Source the tools according to your platform (Windows: double click the
  tools icon; Mac: ``source /Applications/XMOS_XTC_<PATH>/Setenv`` Linux:
  ``pushd <PATH-TO-TOOLS>; source SetEnv; popd``)

* Run::

    ``xmake``

* Plug an explorer board into your computer and run::

    ``xrun --xscope bin/app.xe``

* This should run the network

More info regarding the generated C++ model files
----------------------------

The code is compiled to C++. The compiled code will require the
Tensorflow Lite for Micro run time support. You need to, in your sandbox,
obtain the ``lib_tflite_micro`` module
<https://github.com/xmos/lib_tflite_micro>, which will pull in all other
required modules.

Simply copy the ``model.tflite.cpp`` and ``model.tflite.h`` file to the source
directory of your application, and you can now, from C++ call the following
functions:

* ``model_init(void *flash_data)`` This takes a single parameter, which is a channel end to
  the flash server

* ``model_input_ptr(int index)`` This returns a pointer to the data where
  the input tensor is stored; index should be set to zero unless there are
  multiple inputs.

* ``model_invoke()`` This runs an inference

* ``model_output_ptr(int index)`` Analogous to the output pointer. Note
  that the input may have been overwritten.
  
Integration with sensors
------------------------

There are many sensor interfaces, we will soon publish example programs to
interface to PDM microphones and MIPI/SPI cameras