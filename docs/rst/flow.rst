Work Flow
=========

``xmos-ai-tools`` is available on https://pypi.org/project/xmos-ai-tools/.
It includes:

* the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
* the XCore tflm interpreter to run the transformed models on host

Installation steps
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

  Installing ``xmos-ai-tools`` will make the xcore-opt binary available in your shell to use directly.

* Obtain the tool-chain from http://www.xmos.ai/tools and install it according to the platform instructions.

* Setup ``XMOS_AITOOLSLIB_PATH`` environment variable. This is used to identify the installed location of xmos-ai-tools library and headers.

  On Windows, run the following command::

    FOR /F "delims=" %i IN ('python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))"') DO set XMOS_AITOOLSLIB_PATH=%i

  On MacOS and Linux, run the following command::

    export XMOS_AITOOLSLIB_PATH=$(python -c "import xmos_ai_tools.xinterpreters.device as device_lib; import os; print(os.path.dirname(device_lib.__file__))")



Example applications
----------------------------

These are 5 example models; in order of complexity

* `app_no_flash <../../examples/app_no_flash/README.rst>`_  - a single model, no flash memory used. This is the
  fastest but most pressure on internal memory.

* `app_flash_single_model <../../examples/app_flash_single_model/README.rst>`_ - a single model, with learned parameters in
  flash memory. This removes a lot of pressure on internal memory.

* `app_flash_two_models <../../examples/app_flash_two_models/README.rst>`_ - two models, with learned parameters in flash memory.

* `app_flash_two_models_one_arena <../../examples/app_flash_two_models_one_arena/README.rst>`_ - two models, with learned parameters in
  flash memory. The models share a single tensor arena (scratch memory).



More info regarding the generated C++ model files
----------------------------

The model code is compiled to C++ source and header.
The generated header file contains the simple API to interact with the model.
Some of the commonly used functions are:

* ``model_init(void *flash_data)`` This takes a single parameter, which is a channel end to
  the flash server.

* ``model_input_ptr(int index)`` This returns a pointer to the data where
  the input tensor should be stored; index should be set to zero unless there are
  multiple inputs.

* ``model_invoke()`` This runs an inference

* ``model_output_ptr(int index)`` This returns a pointer to the data where
  the output tensor would be stored.
  
Integration with sensors
------------------------

There are many sensor interfaces, we will soon publish example programs to
interface to PDM microphones and MIPI/SPI cameras
