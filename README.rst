Documentation
=============

Index
-----

* `Run models on XCORE.AI <docs/rst/flow.rst>`_
* `Run models via Python on host <#using-xmos-ai-tools-from-python>`_
* `Examples <examples/README.rst>`_
* `Graph transformer command-line options <docs/rst/options.rst>`_
* `Transforming Pytorch models <docs/rst/pytorch.rst>`_
* `FAQ <docs/rst/faq.rst>`_
* `Changelog <docs/rst/changelog.rst>`_
* Advanced topics

    * `Detailed background to deploying on the edge using XCORE.AI <docs/rst/xcore-ai-coding.rst>`_
    * `Building the graph transformer and xmos-ai-tools package <docs/rst/build-from-source.rst>`_


Installing xmos-ai-tools
------------------------

``xmos-ai-tools`` is available on `PyPI <https://pypi.org/project/xmos-ai-tools/>`_.
It includes:

* the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
* the XCore tflm interpreter to run the transformed models on host


Perform the following steps once:

.. code-block:: shell

    # Create a virtual environment with
    python3 -m venv <name_of_virtualenv>

    # Activate the virtual environment
    # On Windows, run:
    <name_of_virtualenv>\Scripts\activate.bat
    # On Linux and MacOS, run:
    source <name_of_virtualenv>/bin/activate

    # Install xmos-ai-tools from PyPI
    pip3 install xmos-ai-tools --upgrade

Use ``pip3 install xmos-ai-tools --pre --upgrade`` instead if you want to install the latest beta version.

Some older pre-release wheels may expect the ``opcode2name`` helper in the old ``tflite`` package layout.
If you use the host interpreter with one of those wheels and see an ``opcode2name`` import error,
either upgrade to a newer ``xmos-ai-tools`` build containing the compatibility fix,
or constrain ``tflite`` with ``pip3 install "tflite>=2.4.0,<=2.10.0"``.

.. _using-xmos-ai-tools-from-python:

Using xmos-ai-tools from Python
-------------------------------

.. code-block:: python

    from xmos_ai_tools import xformer as xf

    # Optimizes the source model for xcore
    # The main method in xformer is convert, which requires a path to an input model,
    # an output path, and a list of configuration parameters.
    # The list of parameters should be a dictionary of options and their values.
    #
    # Generates -
    #   * An optimized model which can be run on the host interpreter
    #   * C++ source and header which can be compiled for xcore target
    #   * Optionally generates flash image for model weights
    xf.convert("source model path", "converted model path", params=None)

    # Returns the tensor arena size required for the optimized model
    # Only valid after conversion is done
    xf.tensor_arena_size()

    # Prints xformer output
    # Useful for inspecting optimization warnings, if any
    # Only valid after conversion is done
    xf.print_optimization_report()

    # To see all available parameters
    # To see hidden options, run `print_help(show_hidden=True)`
    xf.print_help()

For example:

.. code-block:: python

   from xmos_ai_tools import xformer as xf

   xf.convert("example_int8_model.tflite", "xcore_optimised_int8_model.tflite", [
       ("xcore-thread-count", "5"),
   ])

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-weights-file" option.

.. code-block:: python

   xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", [
       ("xcore-weights-file ", "./xcore_params.params"),
   ])

Some of the commonly used configuration options are described `here <docs/rst/options.rst>`_

Running the xcore model on host interpreter
-------------------------------------------

.. code-block:: python

   from xmos_ai_tools.xinterpreters import TFLMHostInterpreter

   input_data = ... # define your input data

   ie = TFLMHostInterpreter()
   ie.set_model(model_path='path_to_xcore_model', params_path='path_to_xcore_params')
   ie.set_tensor(ie.get_input_details()[0]['index'], value=input_data)
   ie.invoke()

   xformer_outputs = []
   num_of_outputs = len(ie.get_output_details())
   for i in range(num_of_outputs):
       xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]['index'])).

   # Note: use ie.close() or "with TFLMHostInterpreter() as ie:" to free resources
   ie.close()
