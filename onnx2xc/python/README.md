Instructions for using onnx2xc.py

**This is a prototype tool currently meant only for evaluation**
**It is not production quality and may never be.  Proceed with caution!!!**

# Running

    > ./onnx2xc.py -i /path/to/model.onnx -o mnist

This will generate two source files, mnist.h & mnist.c

# Adding a New Operator

1. Implement your new operator in a source file named something like, MyOp.py.  See the existing operators for examples.
2. Add a new `elif` condition for MyOp to the `create()` method in \_\_init\_\_.py.