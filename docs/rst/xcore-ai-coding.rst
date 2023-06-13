Deploying inference on the edge using XCORE.AI
==============================================

This document gives an introduction how to use trained models on XCORE.AI.
XCORE.AI is a cross-over processor by XMOS, capable of real-time IO, and
fast execution of neural networks.

Model suitability
-----------------

The first question that you will have is "can I run my model on your
device". The answer depends on the model and the economic constraints of
the hardware that you wish to deploy the model on. We first explain the
variables of the model, then the characteristics of the XCORE device, and
then some rules of thumb

Model characteristics
+++++++++++++++++++++

Not all models are suitable to be ran on the edge, or indeed execute on
XCORE.AI. In general a model is characterised by the amount of memory that
is needed, the amount of compute, and the type of compute.

Memory required:

* The number of learned parameters, also known as the number of
  coefficients in the network. A small number is beneficial to run on the
  edge, but a larger number may be required to achieve the desired
  inferencing accuracy.

* The space required for the *tensor arena*. The tensor arena is the
  scratch memory that is needed to store the temporary values used by the
  network, whilst evaluating the network. The tools that we provide will
  help you by calculating this number.

* The size of the model excluding the learned parameters. This stores
  information on how the graph is connected, quantisation parameters,
  tensors sizes, etc. This is typically tens of kilobytes. It is linear in
  the number of operators in the model

Compute required:

* The number of operations that a model requires, this is typically
  expressed as the number of multiply-accumulate operations.

Type of compute:

* The operators used in your network. Operators are the basic instructions
  that are performed during inferencing, and may be operations such as
  2D-convolutions, addition, or post-processing-detect.

* The data types used in the model, this may be floats, half floats, int8,
  etc. Typically models are trained using floating point arithmetic, but in
  order to save on memory requirements and memory bandwidth they are often
  *quantized* to 8-bit coefficients and values.

In addition, there may be a required set of IO devices (sensors, actuators)
that surround the model.


XCORE.AI characteristics
++++++++++++++++++++++++

XCORE.AI is an embedded device, and as such there are constraints as to the
size an complexity of the models that can be executed. Multiple xcore.ai
devices can be connected to each other to create larger systems, and
external memory can be connected to increase the usable memory size.
However, there is a limit to what is economical to use for a certain problem.
A single XCORE.AI device comprises:

* Two processors, each with their own memory, each capable of executing
  up to 600 or 800 MHz depending on the version of the device.

* Each processor has 512 kByte of memory (0.5 MByte) which is used for
  storing code, model, and/or parameters

* Each processor can execute up to 32 multiply accumulates per
  clock cycle (on int8), for a total absolute maximum of 51.2 GMacc/s

The device typically needs at least one flash device to store permanent
data (eg, code to boot from, model parameters etc), capable of storing 2-16
MByte of data. In some situations the flash can be avoided if another
processor in the system can serve as a boot-server.

An XCORE.AI device may be connected to an external LPDDR memory, capable of storing
64-128 MByte of data. Multiple XCORE.AI devices may be connected to each other,
scaling memory and performance.

XCORE.AI devices can connect to MIPI cameras (up to 190 Mpixel/s at 8-bit
pixels), MEMS microphones (as many as the system needs, typically 1-8), SPI
radar devices, or other sensors. A wide variety of protocols are available
to communicate with devices or actuators.

What models fit on XCORE.AI?
++++++++++++++++++++++++++++

There is no hard limit on the number of parameters for a model, but larger
models do slow down. Very small models (say, 100 kByte) are held in
internal memory, larger models are typically streamed from flash. The
latter enables networks with up to a few million learned parameters to
execute efficiently. Significantly larger networks, with tens of millions
of learned parameters will can be executed from flash, but the speed of
execution (frame rate) may be lower than desired. Large models may be
stored in external LPDDR memory which is fast but has a slightly higher BOM
cost.

The size of the tensor arena can be a bottle-neck. For optimal execution,
the tensor arena is held in internal memory, which limits the tensor arena
to a few hundred kbytes in size. The XMOS AI tools have methods of
minimising the required tensor arena. Larger tensor arenas can be stored in
external memory, if the BOM cost can sustain adding an external LPDDR
memory. This removes most limits on the size of the tensor arena.

The thurd part that needs to be stored somewhere is the architecture of the
model. This can either go in internal or external memory; it cannot be
stored in flash.

Finally memory is also required to store the code that executes the
operators. The size of the code is around 100-200 kByte - depending on the
number of different operators that are used, whether compilation is used
(less memory) or a full interpreter (more memory), and whether code other
than AI code is present.

We support all virtually operators and types that are in TensorFlow Lite
for Micro; but only a subset of those operators have been optimized to run
on the device. Convolutional networks with int8 datatypes typically run at
high speed. It is fine for some operations to execute as float32. As long
as the very large convolutions use int8 encodings the model is typically
executed efficiently. We do support dense layers, but the nature of dense
layers means that every multiplication needs its own learned parameter
which means that the execution speed of large dense layers may be limited
by the speed of backing
store where the parameters come from.

In addition to the common types we support highly efficient execution of
1-bit networks (also known as binarised neural networks or XNOR networks).
These can execute at a rate of up to 256 GMacc/s.

Needless to say, this creates a very large number of options, from very
large systems to very small systems. We show the use cases of a few systems
below.

XCORE.AI with 64Mbyte LPDDR memory
``````````````````````````````````

This system has very little contraints. It can execute large models that
are held in external memory, and the tensor arena can either be held in
external or internal memory.

If the tensor arena is held in external memory, then the total sum of the
tensor arena and the model should not exceed 64 Mbyte; for example, you may
have 60 MByte of parameters, and 4 MByte of tensor arena.

If the tensor arena is small enough (less than a few hundred kilobytes) then
you can store the tensor arena in internal memory and the model in external
memory. The advantage of this configuration is that it is faster, and you
can use the whole external memory for parameters. 

Having an external memory is a good evaluation system for an initial
network, and it can be used to run large networks before they are optimized
down into smaller networks.

XCORE.AI without external memory
````````````````````````````````

Without external memory, there are fewer options as to where the store the
model and the data. In particular, the tensor arena must be stored in
internal memory, and is therefore limited to around a few hundred kilobytes. The
model can either be stored in internal memory too or in flash memory.

Storing the learned parameters in internal memory reduces the amount of memory available for
the tensor arena, but it is the fastest and most low power way to execute a
model. Storing the learned parameters in flash will result in a slower execution, but
will leave all of internal memory available for the tensor arena. As flash
cannot be written efficiently it cannot be used for the tensor arena.

Assuming that the learned parameters are stored in flash, that means that
the internal memory will be shared between code (instruction sequences
implementing the operators), the tensor arena, and the model architecture.
These three should sum up to no more than 512 kByte.

Using multiple processors
`````````````````````````

In XCORE.AI each processor has 512 kBytes of memory; that means that there
are various ways in which the model can be split over two or more
processors. Examples of splits are:

* A problem that requires more than one model, may execute one model on
  each tile

* A model can be split in a first and second part, with each part running
  on a processor. It may be that the split is organised so that one part
  of the model needs a large tensor arena with a small number of
  parameters, and the second part needs a small tensor arena with many
  parameters.

* A model may be split into a left and a right half, where each half
  occupies a processor. This means that each processor only stores part
  of the tensor arena. The current version of xcore-opt has no automated
  support for this.

How to prepare a network for XCORE.AI
-------------------------------------

The general approach to encoding a problem that incorporates a trained
network on an XCORE.AI chip is as follows:

  #. You train your network as normal, using for example Keras.

  #. You quantize your network to ``int8`` and convert it to TensorFlow
     Lite. You can keep the occasional float operation in the network.

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

  * An *xcore.ai run time*. 

The xcore transformer, compiler, and run-time support can all be installed
with a single pip command: . They can be used
through a python interface or from the command line as required.


Operators
---------

Virtually all tensorflow-lite-for-micro operators are supported
with the exception of Variables, While, and If. Only very few operators
have been optimised to run efficiently on XCORE; those that typically
account for 99% of the execution time. 

Optimized operators
+++++++++++++++++++

The following operators can be optimized by the xcore optimizer into an
equivalent faster or more memory efficient operator:

* Conv2D

* Conv2DDepthwise

* AvgPool2D

* Add

* Concatenate

* Pad

* StridedSlice

* Tanh, Sigmoid, Hardswish, Relu

We are always interested to know of operators that take on a large
proportion of your model.

Constraints on operators
++++++++++++++++++++++++

Some xcore optimizable operators have constraints on them which dictate
situations where they can not be optimized. In particular:

* Make sure that each convolution outputs a multiple of FOUR channels.

* For optimal speed, the number of input channels should be a multiple of
  16, otherwise 4.

* For a first image convolution that typically has three channels (YUV,
  RGB), the graph transformer will insert a fast pad from three to four.

* For a convolution, execution is fast where the bias term of is reasonably
  close to zero.

Worked example
--------------

Github contains two python notebooks that show the whole process:

<https://github.com/xmos/ai_tools/blob/develop/docs/keras_to_xcore.ipynb>
and
<https://github.com/xmos/ai_tools/blob/develop/docs/optimise_for_xcore.ipynb>.

