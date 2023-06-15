Using xformer from Python
=========================

The Python interface "xmos-ai-tools" available through PyPi contains the xcore 
optimiser (xformer) for optimising suitable tflite models. This module can be imported
using:

.. code-block:: Python

  from xmos_ai_tools import xformer

The main method in xformer is convert, which requires an path to an input model,
an output path, and a list of parameters. The list of parameters should be a dictionary
of options and their value. 

.. code-block:: Python

  xf.convert("example_int8_model.tflite", "xcore_optimized_int8_model.tflite", 
    {"mlir-disable-threading": None, "xcore-reduce-memory": None,}
  )

The possible options are described below in the command line interface section. If the default operation is intended this third argument can be "None".

.. code-block:: Python

  xf.convert("source model path", "converted model path", params=None)


To see all available configuration options.

.. code-block:: Python

  from xmos_ai_tools import xformer as xf
  xf.print_help()

This will print all options available to pass to xformer. To see hidden options, run ``print_help(show_hidden=True)``.

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-flash-image-file" option.

.. code-block:: Python

  xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", {
      "xcore-flash-image-file ": "./xcore_params.params",
  })

To combine these files created by the code above into a flash image .out file, use the generate_flash() function

.. code-block:: Python

  from xmos_ai_tools import xformer as xf
  xf.generate_flash(
      output_file="xcore_flash_binary.out",
      model_files=["model.tflite", "model2.tflite"],
      param_files=["1.params", "2.params"]
  )

The flash image .out file can be flashed on XCORE.AI using ``xflash``::

  xflash --data xcore_flash_binary.out --target XCORE-AI-EXPLORER


The python interface also contains a host-side interpreter for tflite
model. This interpreter can be imported from xmos_ai_tools
as follows:

.. code-block:: Python

  from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter

The interface follows the interface of the tensorflow lite interpreter. To
run inference on a model the interpreters can be used as such:

.. code-block:: Python

  ie = xcore_tflm_host_interpreter()
  ie.set_model(model_path = xcore_model)
  ie.set_input_tensor(data = input)
  ie.invoke()

  xformer_outputs = []
  for i in range(num_of_outputs):
      xformer_outputs.append(ie.get_output_tensor(output_index = i))
