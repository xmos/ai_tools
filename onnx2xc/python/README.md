<span style="color:red; font-size:16pt">
This is a prototype tool currently meant only for learning about and evaluating ONNX. It is not production quality and may never be.  Proceed with caution!!!
</span>

Instructions for using onnx2xc.py

# Example model

Only the following model has been tested:

https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz

# Running

    > ./onnx2xc.py -i /path/to/model.onnx -o mnist

This will generate two source files, mnist.h & mnist.c

# Adding a New Operator

1. Implement your new operator in a source file named something like, MyOp.py.  See the existing operators for examples.
2. Add a new `elif` condition for MyOp to the `create()` method in \_\_init\_\_.py.

# Tasks & Future Considerations

* Implement the actual Operators (float32 & int8), stub them out in lib_ai or lib_dsp if necessary.
* Check the the generated code can compile and run using `xsim`.
* Class for an intermediate graph data structure?  This would make it a bit easier to support additional formats.  Each format would be parsed into the intermediate graph struct.  Ed has code to parse TensorFlow in lib_keyword, Ross has code to parse MXNet in lib_ai.
* Add option for benchmarking instrumentation
* Use `#define` for tensor shape dimensions?
* Wrap lines?
* Ouput build files?  Probably not but should think about.
* Address other TODO and FIXME comments sprinkled about in the source.  
