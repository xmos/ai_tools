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

The xformer module contains two more useful methods, "xformer.printf_help()" which will
print information of usage of xformer.convert, and "xformer.generate_flash" useful when
writing models to flash.

If a model is split into a tflite model for flash, and a .params file using
the xformer option "xcore-flash-image-file", the generate_flash method can
be used to combine these files into a binary to be stored in flash on an
xcore.

.. code-block:: Python
  xf.generate_flash("xcore_optimized_int8_flash_model.tflite",  "xcore_params.params", "xcore_flash_binary.out")


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
