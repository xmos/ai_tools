Graph transformation options
============================

The graph transformer can be controlled through options. For example, one
cans use options to set the level of parallelism to use or to specify the
name of the output file.

The method of providing options depends on how the graph transformer is
invoked; whether it is invoked from a Python environment or from a
command-line environment. In both cases, the same options can be provided.
Below, we first describe how to pass options from Python and command-line
arguments respectively, then a list of most-used options, and then a list
of all options.

Transformation options in a Python environment
----------------------------------------------

The Python interface "xmos-ai-tools" available through PyPi contains the xcore 
optimiser (xformer) for optimising suitable tflite models. This module can be imported
using:

.. code-block:: Python

  from xmos_ai_tools import xformer

The main method in xformer is convert, which requires an path to an input model,
an output path, and a list of parameters. The list of parameters should be a dictionary
of options and their value. 

.. code-block:: Python

  xf.convert("example_int8_model.tflite", "xcore_optimized_int8_model.tflite", 
             params = {
                "xcore-thread-count": 4,
                "xcore-reduce-memory": None,
             }
            )

The possible options are described below in the command line interface section. If the default operation is intended this third argument can be "None".

.. code-block:: Python
  
  xf.convert("example_int8_model.tflite", "xcore_optimized_int8_model.tflite", 
             params = None
  )


Transformation options in a command-line environment
----------------------------------------------------

Upon installing the "xmos-ai-tools" from PyPi, the program ``xcore-opt`` is
available on the command-line. It is called with at least one argument (the
input model), and all options are specified with a ``--`` ahead of it, eg::

  xcore-opt example_int8_model.tflite --xcore-thread-count 4 --xcore-reduce-memory


Options
-------

``xcore-thread-count N``
++++++++++++++++++++++++

Number of threads to translate for. Defaults to 1.


``xcore-flash-image-file filename``
+++++++++++++++++++++++++++++++++++

File to place the learned parameters in. If this option is not specified,
the learned parameters are kept by the model. This will increase the amount
of RAM required by the model but is very fast. When this option is used,
the learned parameters are placed in a file that must be flashed onto the
hardware, and the learned parameters will be streamed from flash. This can
be slower but allows large numbers of learned parameters to be used.

``xcore-load-externally-if-larger N``
+++++++++++++++++++++++++++++++++++++

Sets a threshold under which to not place learned parameters in flash. The
default is set to 96. This option is only meaningful if
``xcore-flash-image-file`` has been used. You can experiment with this
parameter to get a different trade-off between speed and memory requirements.
                          
``xcore-reduce-memory true|false``
++++++++++++++++++++++++++++++++++

Try to reduce memory usage by possibly increasing
execution time. Default is true

``xcore-conv-err-threshold R``
++++++++++++++++++++++++++++++

When optimising convolutions small inaccuracies are introduced, due to the
nature of fixed point comitations. These errors are typically small and
happen infrequently. The default threshold is 0.25, meaning that the
largest error that is acceptable is two bits below the decimal comma (in
integer arithmetic). If an error higher than this occurs, the compiler will
fall back on a less optimal convolution that produces a better result.

You can adjust this parameter to get a different trade-off between
execution speed and accuracy of the result.


Hmmmmmm - CL and Python are different?
++++++++++++++++++++++++++++++++++++++

* ``-o filename.tflite``        Name of the file where to place the optimized
                          TFLITE flatbuffer


Advanced options
----------------

``xcore-force-conv-err-full-check``
+++++++++++++++++++++++++++++++++++

By default a the above option calculates an upper-bound for the error.
Setting this option calculates the precise maximum error at the cost of
(significant) extra compile-time.

``xcore-conv-multiplier-factor``
++++++++++++++++++++++++++++++++
  
There are networks where large errors in a layer can be fixed by changing
the quantization. This option limits outliers in the multipliers of a
convolution to a factor of N larger than the minimum. THe default for N is
0x7fffffff (ie, no limit).
                          
``xcore-dont-minify``
+++++++++++++++++++++

Normally the TFLITE model is minified, by reducing string lengths, using
this option enables you to keep the old strings.
