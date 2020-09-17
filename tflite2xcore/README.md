# Converting TensorFlow Lite model to xCORE

    > ./tflite2xcore_graph_conv.py path/to/your_model.tflite path/to/your_converted_model.tflite

# Cenerating C code for xCORE TensorFlow Lite model

    > ./tflite2xcore_code_gen.py --name your_model path/to/your_model.tflite

# Visualizing a Model

    > ./tfite_visualize.py path/to/your_model.tflite -o your_model.html 

# Adding a New Operator to Code Generation

1. Implement your new operator in a source file named something like, MyOp.py.  See the existing operators for examples.
2. Add a new `elif` condition for MyOp to the `create()` method in \_\_init\_\_.py.
