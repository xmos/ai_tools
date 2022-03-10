# XMOS AI Tools

## Usage

### Using xformer
```python
from xmos_ai_tools import xformer as xf

xf.convert("source model path", "converted model path", params=None)
```
where `params` is a dictionary of compiler flags and paramters and their values.

For example:
```python
from xmos_ai_tools import xformer as xf

xf.convert("example_int8_model.tflite", "xcore_optimised_example_int8_model.tflite", {
    "mlir-disable-threading": None,
    "xcore-reduce-memory": None,
})
```

To see all available parameters, call
```python
from xmos_ai_tools import xformer as xf

xf.xformer_help()
```
This will print all options available to pass to xformer. To see hidden options, run `xformer_help(show_hidden=True)`


### Using the xcore tflm host interpreter
```
from xmos_ai_tools import xcore_tflm_host_interpreter as xtflm

ie = xtflm.XTFLMInterpreter(model_content=xformed_model)
ie.set_input_tensor(0, input_tensor)
ie.invoke()
xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_output_tensor(i))
```
