Example without flash
=====================

Please consult `here <../../docs/rst/flow.rst>`_ on how to install the tools.

In order to compile and run this example follow these steps::

  xcore-opt vww_quant.tflite -o model.tflite
  mv model.tflite.cpp model.tflite.h src
  xmake
  xrun --xscope bin/app_no_flash.xe

This should print around 300 lines ending with::

  [...]
  Human (98%)

The lines printed are profiling information which we will explain in detail
below. Profiling is enabled by this line in the Makefile::

  APP_FLAGS += -DTFLMC_XCORE_PROFILE

We show how profiling can be used to reduce execution time from 74 to 8
milli-seconds.

Purpose of profiling
--------------------

The purpose of profiling is for you to understand how fast the network
runs, and which parts of the network are taking how much time. If some
parts of the network are taking too much time, they may be remedied by
changing the network, or by suggesting an optimisation for this type of
network to the team at XMOS.

How to enable profiling
-----------------------

Profiling is controlled by a ``#define``. By defining the
``TFLMC_XCORE_PROFILE`` symbol profiling will be printed out.

Interpreting the profiling output
---------------------------------

Profiling output comprises nine sections:

* Per node, per class, and total time for the initialisation,
* Per node, per class, and total time for the preparation, and
* Per node, per class, and total time for the inference

Typically, the last two are the most interesting.

All times are printed in 10 ns ticks (0.000,000,01 second), and summarised
in milliseconds.

* Per node profiling lists the number of tick for each node in the graph
  (eg, a Convolution or a Sigmoid)

* Summaries add together all nodes of the same type, and show the total
  number of ticks and milliseconds for each class of operator.

* The total time lists the total time for a stage.

* Initialisation and preparation are two actions that prepare the graph and
  the tensor-arena for the inferencing step. They are typically executed
  once before many inferences are ran. For example, for a keyword spotter
  one would initialise and prepare the tensor arena once, and then
  inference time and time again.

* Inferencing time is the time taken to, given some input data, infer the
  output.

Note that the times we give below may be different depending on your
target, the precise version of the AI tools, and the version of the
underlying compiler.

The first three sections may be::

  Profiling init()...
  node 0     OP_XC_pad_3_to_4                 320         
  node 1     OP_XC_pad                        536
  [...]
  node 44    OP_XC_strided_slice              1555        
  node 46    OP_SOFTMAX                       21          
  
  Cumulative times for init()...
  OP_XC_pad_3_to_4                 320          0.00ms
  OP_XC_pad                        7503         0.08ms
  OP_XC_conv2d_v2                  90276        0.90ms
  OP_DEPTHWISE_CONV_2D             42           0.00ms
  OP_XC_strided_slice              1555         0.02ms
  OP_RESHAPE                       0            0.00ms
  OP_SOFTMAX                       21           0.00ms
  
  Total time for init() - 99717      1.00ms

The first lines lists initialisation per node. 320 ticks (3.2 us) for node
0, 536 ticks (5.36 us) for node 1, etc. The second section shows that all
OP_XC_PAD operators together took 7503 ticks (75 us) to initialise. The
final section shows that initialisation all together took 1 ms.

In a similar vein, the next three sections is the preparation of the tensor
arena, which takes a total of 1.71 ms.

The last three sections are as follows::

  node 0     OP_XC_pad_3_to_4                 26991       
  node 1     OP_XC_pad                        13945       
  [...]
  node 40    OP_DEPTHWISE_CONV_2D             2243362     
  node 41    OP_XC_conv2d_v2                  34557       
  node 42    OP_XC_conv2d_v2                  2550        
  node 43    OP_XC_conv2d_v2                  591         
  node 44    OP_XC_strided_slice              106         
  node 45    OP_RESHAPE                       131         
  node 46    OP_SOFTMAX                       1663        


  Cumulative times for invoke()...
  1     OP_XC_pad_3_to_4                 26991        0.27ms
  14    OP_XC_pad                        53359        0.53ms
  27    OP_XC_conv2d_v2                  2916236      29.16ms
  2     OP_DEPTHWISE_CONV_2D             4487819      44.88ms
  1     OP_XC_strided_slice              106          0.00ms
  1     OP_RESHAPE                       131          0.00ms
  1     OP_SOFTMAX                       1663         0.02ms
  
  Total time for invoke() - 7486305    74.86ms

The bottom line shows the total time for the invokation of this network:
74.86ms; the lines above it shows in detail where the time went, note that
the two main parts of the inferencing are::

  27    OP_XC_conv2d_v2                  2916236      29.16ms
  2     OP_DEPTHWISE_CONV_2D             4487819      44.88ms

The first number of each line is the number of times that the operator was
invoked. The second column is the operator name, the third column the
number of ticks, and the final column the time in milli-seconds.

The ``OP_XC_conv2d_v2`` operator executed 27 times, so on average took just
over one millisecond per invokation. The ``OP_DEPTHWISE_CONV_2D`` operator
took 22.44 ms per invokation. The reason this operator is so slow is
twofold. Any operator that has ``XC`` (for XCORE) in the name is an
optimised operator; this is a reference implementation. The reason that the
reference implementation was used is because of a warning emitted by the
graph transformer::

  vww_quant.tflite:0:0: ... Quantization error of 0.325626 larger ... reverting to reference DepthwiseConv2D op

The converter identified that the reference operator had a much higher
accuracy than the optimised operator, and it reverted to the reference
operator. As a first test we can change the threshold for this to happen::

  xcore-opt vww_quant.tflite -o model.tflite --xcore-conv-err-threshold=0.5
  mv model.tflite.cpp model.tflite.h src
  xmake
  xrun --xscope bin/app_no_flash.xe

Moving the source files, recompiling and executing it yields the following
profiling output::

  Cumulative times for invoke()...
  1     OP_XC_pad_3_to_4                 26993        0.27ms
  14    OP_XC_pad                        54115        0.54ms
  29    OP_XC_conv2d_v2                  2971789      29.72ms
  1     OP_XC_strided_slice              110          0.00ms
  1     OP_RESHAPE                       134          0.00ms
  1     OP_SOFTMAX                       1668         0.02ms
  
  Total time for invoke() - 3054809    30.55ms


This shows that the total execution time was reduced by 44 milliseconds.
The optimised convolution was executed twice more, accounting for an extra
556 micro-seconds; the optimised operator is 80 times faster than the
reference implementation.

The final optimisation is to parallelise the execution. We do this by
telling the graph-transformer to generate code that uses five threads::

  xcore-opt vww_quant.tflite -o model.tflite --xcore-conv-err-threshold=0.5 --xcore-thread-count=5
  mv model.tflite.cpp model.tflite.h src
  xmake
  xrun --xscope bin/app_no_flash.xe

Moving the files, recompiling, and rerunning the code yields::

  Cumulative times for invoke()...
  1     OP_XC_pad_3_to_4                 26993        0.27ms
  14    OP_XC_pad                        54115        0.54ms
  29    OP_XC_conv2d_v2                  717748       7.18ms
  1     OP_XC_strided_slice              109          0.00ms
  1     OP_RESHAPE                       134          0.00ms
  1     OP_SOFTMAX                       1668         0.02ms
  
  Total time for invoke() - 800767     8.01ms

Execution time has gone down from 30.55 to 8 milli-seconds. Five threads
has only yielded a 3.8x speedup on this occasion; typically on larger
graphs the speedup is closer to 5x.
