# Converting TensorFlow Lite model to xCORE

    > ./tflite2xcore_graph_conv.py path/to/your_model.tflite path/to/your_converted_model.tflite

# Cenerating C code for xCORE TensorFlow Lite model

    > ./tflite2xcore_code_gen.py --name your_model path/to/your_model.tflite

# Visualizing a Model

    > ./tfite_visualize.py --html_output your_model.html --no_browser path/to/your_model.tflite

# Adding a New Operator

1. Implement your new operator in a source file named something like, MyOp.py.  See the existing operators for examples.
2. Add a new `elif` condition for MyOp to the `create()` method in \_\_init\_\_.py.

# Tasks & Future Considerations

* Class for an intermediate graph data structure?  This would make it a bit easier to support additional formats.  Each format would be parsed into the intermediate graph struct.  Ed has code to parse TensorFlow in lib_keyword, Ross has code to parse MXNet in lib_ai.
* Add option for benchmarking instrumentation
* Use `#define` for tensor shape dimensions?
* Wrap lines?
* Ouput build files?  Probably not but should think about.
