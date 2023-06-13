Work Flow
=========


Initial steps
-------------

Perform the following steps once:

* ``pip3 install xmos-ai-tools``; use a virtual-environment of your choice.
  
* Obtain the tool-chain from http://www.xmos.ai/tools and install it
  according to the platform instructions.

* Create a sandbox (working directory)

* In this sandbox unpack lib_tflite_micro ``git clone https://github.com/xmos/lib_tflite_micro``

  Inside ``lib_tflite_micro`` execute ``git submodule update --init --recursive``

For each application program
----------------------------

* Create a directory my_models/app_model_1 inside the working directory

* Copy your .TFLITE file to this directory

* Run the graph-transformer::

    xcore-opt model.tflite

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

The API created by the model
----------------------------

The code is compiled to C++. The compiled code will require the
TensorflowLite for Micro run time support. You need to, in your sandbox,
obtain the ``lib_tflite_micro`` module
<https://github.com/xmos/lib_tflite_micro>, which will pull in all other
required modules.

Simply copy the ``<model>.cpp`` and ``<model>.h`` file to the source
directory of your application, and you can now, from C++ call the following
functions [[This needs to be doxygened for consistency]]:

* ``<model>_init(void *flash_data)`` This takes a single parameter, which is a channel end to
  the flash server

* ``<model>_input_ptr(int index)`` This returns a pointer to the data where
  the input tensor is stored; index shoudl be set to zero unless there are
  multiple inputs.

* ``<model>_invoke()`` This runs an inference

* ``<model>_output_ptr(int index)`` Analogous to the output pointer. Note
  that the input may have been overwritten.
  
Integration with sensors
------------------------

There are many sensor interfaces, we will soon publish example programs to
interface to PDM microphones and MIPI/SPI cameras
