Testing ai_tools, lib_nn, lib_tflite_micro, and aisrv
=====================================================

[Work in progress - edit to your heart's content]

This document describes the testing philosophy behind the repos above. Since ai_tools takes
a long time to build we use three CI procedures:

  #. ai_tools CI involves:

     * checkout
     
       * ai_tools
       
       * lib_nn
       
       * lib_tflite_micro
       
     * building xformer in ai_tools
     
     * On x86 build a test-interpreter.dll in lib_tflite_micro=

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
     
  #. lib_nn/lib_tflite_micro CI involves:

     * checkout
     
       * lib_nn
       
       * lib_tflite_micro
       
       * aisrv
       
     * build an aiserver
     
     * grab the latest xformer artifact that is compatible with lib_nn and lib_tflite_micro
     
     After that we systematically test 
     lib_tflite_micro and lib_nn:
     
     * foreach test in large-table-of-lib_tflite_micro-and-lib_nn-tests:
     
       * run the xformer over the test-model
       
       * start aiserver on a piece of hardware
       
       * load the model, run an inference on the input and check the output using the xcore_ai_ie API
       
     This process indicates that any XCORE specific parts of lib_tflite_micro and lib_nn are correct.
     It assumes that XFORMER and AISERVER are working.
     
  #. aisrv CI involves:

     * building aiserver from scratch
     
     * grab the latest xformer artifact that is compatible with lib_nn and lib_tflite_micro
          
     * foreach test in large-table-of-aisrv-tests:
     
       * run the test on hardware (this may involve xformer, the Python xcore_ai_ie API over USB and or SPI)

     This goes through all the corner cases of the AI-server API.
     Like the previous test, this needs real harwdare.
     
    
