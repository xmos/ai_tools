How to prepare a network for XCORE.AI
=====================================

The general approach to encoding a problem that incorporates a trained
network on an XCORE.AI chip is as follows:

  #. You train your network as normal, using for example Keras.

  #. You quantize your network to ``int8`` and convert it to TensorFlow
     Lite. We also support ``16x8`` where intermediate results occupy 16
     bits and parameters are stored in eight bits.
     You can keep the occasional float operation in the network without
     affecting performance.

  #. You optimize your network for XCORE.AI

  #. You evaluate and deploy your network on XCORE.AI

Several components are being used in this process:

  * A *training framework*. This can be any training framework that is
    available as long as there is a way to produce TensorFlow Lite on the
    output. This may be through, for example, exporting to ONNX.

  * A *quantizer*. The post training quantization step takes your
    network and a set of representative data, and transforms all operators
    to operate on low-precision integers. Rather than operating on floating
    point values (16- or 32-bit floating point numbers), the network will be
    operating on signed bytes (8-bit integers in the range [-128..127]).

    In order to compute an appropriate mapping from floating point values
    to integer values, you need to provide a representative dataset to be used
    during the transformation, and this will ensure that intermediate
    values use the full range of int8 values.

    Typically we use the TensorFlow Lite quantizer to perform this step,
    and the output of this step is a *flatbuffer* that contains the
    architecture of the model and all the coefficients.

  * An *xcore transformer*. It takes a flatbuffer from the previous step,
    and converts it into a flatbuffer that has been optimized for the
    XCORE. Note that this step is not required, and the flatbuffer can be
    executed "as is", but this execution will be painfully slow. The xcore
    transformer simply produces an xcore-specific flatbuffer given a
    generic flatbuffer, using operators optimized for xcore.

  * An *xcore.ai compiler*. It takes a flatbuffer and compiles it to C++, which 
    then be compiled to a binary to be executed on the xcore.

The xcore transformer, compiler, and run-time support can all be installed
with a single pip command: . They can be used
through a python interface or from the command line as required.

