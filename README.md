Documentation
-------------

## Index
- [Run models on XCORE.AI](docs/rst/flow.rst)
- [Run models via Python on host](#using-xmos-ai-tools-from-python)
- [Examples](examples/README.rst)
- [Graph transformer command-line options](docs/rst/options.rst)
- [Transforming Pytorch models](docs/rst/pytorch.rst)
- [FAQ](docs/rst/faq.rst)
- [Changelog](docs/rst/changelog.rst)
- Advanced topics
	- [Detailed background to deploying on the edge using XCORE.AI](docs/rst/xcore-ai-coding.rst)
	- [Building the graph transformer and xmos-ai-tools package](docs/rst/build-from-source.rst)


## Installing xmos-ai-tools

``xmos-ai-tools`` is available on [PyPI](https://pypi.org/project/xmos-ai-tools/).
It includes:

* the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
* the XCore tflm interpreter to run the transformed models on host


Perform the following steps once:

```shell
# Create a virtual environment with
python3 -m venv <name_of_virtualenv>

# Activate the virtual environment
# On Windows, run:
<name_of_virtualenv>\Scripts\activate.bat
# On Linux and MacOS, run:
source <name_of_virtualenv>/bin/activate

# Install xmos-ai-tools from PyPI
pip3 install xmos-ai-tools --upgrade
```
Use ``pip3 install xmos-ai-tools --pre --upgrade`` instead if you want to install the latest development version.

<a name="using-xmos-ai-tools-from-python"></a>
## Using xmos-ai-tools from Python

```python
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

```

For example:
```python
from xmos_ai_tools import xformer as xf

xf.convert("example_int8_model.tflite", "xcore_optimised_int8_model.tflite", {
    "xcore-thread-count": "5",
})
```

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-weights-file" option.
```python
xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", {
    "xcore-weights-file ": "./xcore_params.params",
})
```

Some of the commonly used configuration options are described [here](docs/rst/options.rst)

## Running the xcore model on host interpreter

```python
from xmos_ai_tools.xinterpreters import TFLMHostInterpreter

input_data = ... # define your input data

ie = TFLMHostInterpreter()
ie.set_model(model_path='path_to_xcore_model', params_path='path_to_xcore_params')
ie.set_tensor(ie.get_input_details()[0]['index'], value=input_data)
ie.invoke()

xformer_outputs = []
num_of_outputs = len(ie.get_output_details())
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]['index']))
```
