# XMOS AI Tools

## Usage

### xformer
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

xf.print_help()
```
This will print all options available to pass to xformer. To see hidden options, run `print_help(show_hidden=True)`


### xinterpreters
```python
from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter
from xmos_ai_tools.xinterpreters import xcore_tflm_usb_interpreter
from xmos_ai_tools.xinterpreters import xcore_tflm_spi_interpreter

ie = xtflm.XTFLMInterpreter(model_content=xformed_model)
ie = xcore_tflm_host_interpreter()

ie.set_model(model_path = xcore_model, model_index = 0, secondary_memory = False, flash = False)
#secondary_memory and flash arguments ignored on xcore_tflm_host_interpreter
ie.set_input_tensor(data = input, input_index = 0, model_index = 0)

ie.invoke()

xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_output_tensor(output_index = i, model_index = 0))
```
