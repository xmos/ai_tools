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

xf.print_help()
```
This will print all options available to pass to xformer. To see hidden options, run `print_help(show_hidden=True)`


### Using the xcore tflm host interpreter
```python
from xmos_ai_tools import xcore_tflm_host_interpreter as xtflm

ie = xtflm.XTFLMInterpreter(model_content=xformed_model)
ie.set_input_tensor(0, input_tensor)
ie.invoke()
xformer_outputs = []
for i in range(num_of_outputs):
    xformer_outputs.append(ie.get_output_tensor(i))
```

### Using the Keras Model Validator
This allows you to check whether you have missed any opportunities to slightly alter your model so that xformer can omtimise the model to better run on xcore.

#### Parameters:

- **tensorflow.keras.Model model:** Model which to validate
- **xmos_ai_tools.keras.Strictness.Strictness strictness:** `WARNING` | `ERROR`
    
    `WARNING` to print opportunities to allow for further optimisation to stderr.

    `ERROR` to throw an `XCoreUnoptimisedError` exception if an optimisation opportunity was missed.
- **bool allow_padding:** Whether to throw error or print warning if `padding_indirect` is used.

#### XCoreUnoptimisedError
This is the exception raised if an optimisation opportunity was missed.

- To get the message: `str(e)`
- To get the index of the layer in question: `e.get_layer_idx()`

Example: 
```python
from xmos_ai_tools import keras as xmos_keras_tools

# Make model
input = keras.Input(shape=(28, 28, 4), name="img")
x = layers.Conv2D(filters=32, kernel_size=4, activation="relu", padding="same")(input)
x = layers.Conv2D(filters=32, kernel_size=4, activation="relu")(x)
x = layers.Conv2D(filters=16, kernel_size=4, activation="relu")(x)
output = layers.GlobalMaxPooling2D()(x)
model = keras.Model(encoder_input, encoder_output, name="encoder")

# Run Keras Model through Validator
xmos_keras_tools.validate(model=model, strictness=Strictness.ERROR, allow_padding=False)

# Run Keras Model through Validator, Checking only Conv2D
xmos_keras_tools.validate_conv_2d(model=model, strictness=Strictness.ERROR, allow_padding=False)
```

#### Currently supported layer types:
  - Conv2D
  - DepthwiseConv2D