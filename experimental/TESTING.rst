Interaciton and test of ai_tools, lib_nn, lib_tflite_micro, and aisrv
=====================================================================

[Work in progress - edit to your heart's content]

Interfaces, artifacts, APIs, etc.
---------------------------------

``ai_tools`` consumes and produces flatbuffers storing tflite files at present. At some stage,
it may start emitting C or XCORE code. It has a ever growing interface with ``lib_nn``: for 
each operator there is one or more functions to assist in boggling coefficients for the
operator.

``lib_nn`` contains all the XCORE specific code to implement our NN kernels. That includes
assembly code that runs on the xcore, and compile-time code that manipulates ("boggles")
coefficients to be in the right format for the operator. ``lib_nn`` has two APIs: an API
for the run-time-support system that provides functions for calculating, for example, a
2D convolution; and an API for the compiler/test harness to compute, for example, the 
memory block needed to drive the 2D convolution.

``lib_tflite_micro`` contains a vanilla copy of ``tflite_micro`` and a few XCORE specific
files that implement operators such as ``xcore_conv2d``. ``lib_tflite_micro`` has two APIS.

  * an API ``inference_engine.cc`` that provides a C interface to the inference engine. This C interface
    comprises functions to create an inference engine, load a model, and run an inference. The
    C object encapsulating the interface provides a set of pointers that enables tensors to be set
    and get. ``lib_tflite_micro`` uses ``lib_nn`` to implement XCORE specific operators.

  * an API ``tflm_interpreters`` provides a command line interface to lib_tflite_micro
  
  * an API ``tflite_interpreters`` provides a Python interface to lib_tflite_micro for testing
    purposes
    
``aisrv`` has four interfaces: a USB protocol to drive AI engines, a SPI protocol to drive AI
engines, a host Python interface that encapsulates the USB/SPI protocols, and an embedded
C-interface that encapsulates the SPI protocol. Internally, ``aisrv`` has a sensor interface.

Build dependencies
------------------

When building ``ai_tools`` it just depends on  parts of ``lib_nn``: all boggle functions in lib_nn.

``lib_nn`` is not typically built standalone. It can be built on either XCORE or a vanilla host platform
for testing purposes.

``lib_tflite_micro`` is not typically built standalone. It can be built in order to create command line
and python interfaces to TFLM. ``lib_tflite_micro`` has implicit (submodule) dependencies on four external
repos: ``tflite_micro``, ``ruy``, ``flatbuffers`` and ``gemmlwop``. It has an external dependency on ``lib_nn``

When building ``aisrv`` it depends on ``lib_nn` and ``lib_tflite_micro``. In order to run programs on ``aisrv``
one needs to have an XFORMER artefact.

Test dependencies
-----------------

``ai_tools`` is (will be) tested using ``lib_tflite_micro``, and needs ``lib_tflite_micro`` and ``lib_nn``
as external dependencies. These must both be up to date and the version of ``lib_nn`` in particular must
match the version used when building ``ai_tools``.

``lib_nn`` is tested stand-alone in order to test unit-test each function in lib_nn.

``lib_tflite_micro`` has no unit-tests at present.

``aisrv`` depends on ``lib_nn` and ``lib_tflite_micro`` for its tests, but the precise versions of these
are mostly irrelevant for the stand-alone tests of ``aisrv``. In particular, ``aisrv`` tests do not exercise
any kernels in great detail, they just use two very simple reference operators in ``tflite_micro`` in order
to test the interface between ``aisrv`` and ``lib_tflite_micro``.

Testing
-------

We perform the following tests:

  #. A test that the XFORMER is correct.
     This test confirms that XFORMER produces an approriate graph and tests that graph
     on the host. This test uses lib_tflite_micro and lib_nn in order to run programs on
     a host.
     
  #. A test of the AI-server. This tests that the AI server protocols over USB
     and SPI perform as expected. It requires lib_nn and lib_tflite_micro to be
     compileable, but it will not test them; it assumes that they work.
  
  #. Unit tests on ``lib_nn``. These test that each kernel operates as expected.
  
  #. An integration test of the XCORE specific parts of lib_nn and lib_tflite_micro. Assuming
     that the XFORMER artefacts are up-to-date, this test confirms that the 
     assembly kernels in ``lib_nn`` perform as expected starting from a graph,
     and it confirms that any
     XCORE specific part in lib_tflite_micro performs as expected starting from a graph. In particular,
     this shows that the inference_engine.cc interface performs its duties.
     
     This test relies on an XFORMER artifact, and relies on the AI server being
     functionally correct.
     
  #. Integration test of AI-server and XFORMER...

Continuous Integration
----------------------

This leads us to the following CI processes:

  * If we change xformer: run XFORMER CI, this leaves an XFORMER artefact.

  * If we change AI server: run AI server CI
  
  * If we change lib_nn: run XCORE-specific parts CI, run XFORMER CI.

  * If we change lib_tflite_micro: run XCORE-specific parts CI, run XFORMER CI.

The details
-----------

ai_tools CI
+++++++++++

     * checkout
     
       * ai_tools
       
       * lib_nn
       
       * lib_tflite_micro
       
     * build xformer in ai_tools
     
     * build a test-interpreter.dll in lib_tflite_micro

     * foreach test in large-table-of-ai_tools-tests:
    
        * run the xformer over the test-model
        
        * use the test-interpreter to execute the transformed model on a the given test-input
        
        * compare the output with the expected test-output
     
     This process indicates that XFORMER performs the right transformations, and therefore XFORMER can be released.
     This also indicates that the classes in lib_nn that ai_tools depends on are correct.
     It also indidcates that the operators in lib_tflite_micro are correct.
     This process does *not* test that the XCORE specific parts (assembly code in lib_nn, #ifdef XCORE in lib_tflite_micro) are correct.
     
     On merging, the new XFORMER artifact is uploaded, likely compiled for a variety of platforms.
     
     Note: as the xformer depends on lib_nn this CI should also be ran on any change in lib_nn

aisrv CI
++++++++

     * checkout
     
       * lib_nn
       
       * lib_tflite_micro
       
       * aisrv

     * build aiserver
               
     * foreach test in large-table-of-aisrv-tests:
     
       * run the test on hardware (this may involve the Python xcore_ai_ie or C API over USB and or SPI)

     This goes through all the corner cases of the AI-server API.
     This needs real harwdare.
     
     
lib_nn/lib_tflite_micro CI
++++++++++++++++++++++++++

     * checkout
     
       * lib_nn
       
       * lib_tflite_micro
       
       * aisrv
       
     * build an aiserver
     
     * grab the latest xformer artifact that is compatible with lib_nn and lib_tflite_micro
     
     After that we systematically test lib_tflite_micro and lib_nn:
     
     * foreach test in large-table-of-lib_tflite_micro-and-lib_nn-tests:
     
       * run the xformer over the test-model
       
       * start aiserver on a piece of hardware
       
       * load the model, run an inference on the input and check the output using the xcore_ai_ie API
       
     This process indicates that any XCORE specific parts of lib_tflite_micro and lib_nn are correct.
     It assumes that XFORMER and AISERVER are working.
    
