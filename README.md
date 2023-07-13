Documentation
-------------

## Index
- [How to use the graph transformer to run a sample model app on XCORE.AI](docs/rst/flow.rst)
- [Usage from Python to run a sample model on host](https://github.com/xmos/ai_tools/blob/develop/docs/rst/python.rst)
- [Graph transformer command-line options](https://github.com/xmos/ai_tools/blob/develop/docs/rst/options.rst)
- [Transforming Pytorch models](https://github.com/xmos/ai_tools/blob/develop/docs/rst/pytorch.rst)
- [FAQ](https://github.com/xmos/ai_tools/blob/develop/docs/rst/faq.rst)
- [Changelog](https://github.com/xmos/ai_tools/blob/develop/docs/rst/changelog.rst)
- Advanced topics
	- [Detailed background to deploying on the edge using XCORE.AI](https://github.com/xmos/ai_tools/blob/develop/docs/rst/xcore-ai-coding.rst)
	- [Building the graph transformer and xmos-ai-tools package](https://github.com/xmos/ai_tools/blob/develop/docs/rst/build-xformer.rst)


## Quick intro to xmos-ai-tools

``xmos-ai-tools`` is available on [PyPI](https://pypi.org/project/xmos-ai-tools/).
It includes:

* the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
* the XCore tflm interpreter to run the transformed models on host


Perform the following steps once:

* ``pip3 install xmos-ai-tools --upgrade``; use a virtual-environment of your choice.

  Use ``pip3 install xmos-ai-tools --pre --upgrade`` instead if you want to install the latest development version.

```python
from xmos_ai_tools import xformer as xf

xf.convert("source model path", "converted model path", params=None)
```
where `params` is a dictionary of compiler flags and parameters and their values.

For example:
```python
from xmos_ai_tools import xformer as xf

xf.convert("example_int8_model.tflite", "xcore_optimised_int8_model.tflite", {
    "xcore-thread-count": "5",
})
```

To see all available parameters, call
```python
from xmos_ai_tools import xformer as xf

xf.print_help()
```
This will print all options available to pass to xformer. To see hidden options, run `print_help(show_hidden=True)`

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-flash-image-file" option.
```python
xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", {
    "xcore-flash-image-file ": "./xcore_params.params",
})
```


### Run model on host interpreter

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
