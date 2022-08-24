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

xf.convert("example_int8_model.tflite", "xcore_optimised_int8_model.tflite", {
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

To create a parameters file and a tflite model suitable for loading to flash, use the "xcore-flash-image-file" option.
```python
xf.convert("example_int8_model.tflite", "xcore_optimised_int8_flash_model.tflite", {
    "xcore-flash-image-file ": "./xcore_params.params",
})
```

To use combine these files created by the code above into a .out file use the generate_flash() function
```python
xf.generate_flash("xcore_optimised_int8_flash_model.tflite",  "xcore_params.params", "xcore_flash_binary.out")
```

### xinterpreters

Host Interpreter
```python
from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter

ie = xcore_tflm_host_interpreter()
ie.set_model(model_path = xcore_model)
ie.set_tensor(ie.get_input_details()[0]['index'], data = input)
ie.invoke()

xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]['index']))
```
Device Interpreter (USB)
```python
from xmos_ai_tools.xinterpreters import xcore_tflm_usb_interpreter
from xmos_ai_tools.xinterpreters import xcore_tflm_spi_interpreter

ie = xcore_tflm_usb_interpreter()
ie.set_model(model_path = xcore_model, secondary_memory = False, flash = False)
ie.set_tensor(ie.get_input_details()[0]['index'], data = input)
ie.invoke()

xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_tensor(ie.get_output_details()[i]['index']))
```
