Documentation
-------------

## Index
- [How to use the graph transformer to run a sample model app on XCORE.AI](docs/rst/flow.rst)
- [Graph transformer command-line options](docs/rst/options.rst)
- [Transforming Pytorch models](docs/rst/pytorch.rst)
- [FAQ](docs/rst/faq.rst)
- [Changelog](docs/rst/changelog.rst)
- Advanced topics
	- [Detailed background to deploying on the edge using XCORE.AI](docs/rst/xcore-ai-coding.rst)
	- [Building the graph transformer and xmos-ai-tools package](docs/rst/build-xformer.rst)


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

### Using xformer from Python

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

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-flash-image-file" option.
```python
xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", {
    "xcore-flash-image-file ": "./xcore_params.params",
})
```

Some of the commonly used configuration options are described [here](docs/rst/options.rst)

### Running the xcore model on host interpreter

```python
from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter

ie = xcore_tflm_host_interpreter()
ie.set_model(model_path='path_to_xcore_model', params_path='path_to_xcore_params')
ie.set_tensor(ie.get_input_details()[0]['index'], value='input_data')
ie.invoke()

xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]['index']))
```
