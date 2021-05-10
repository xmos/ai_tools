Build instructions
--

`Bazel` is used to build this program. The choice of build system is
driven by the most complex dependency (Tensorflow). 

With Bazel installed (check `.bazelversion` for current version),
you can build with the following command:

    bazel build //:xcore-opt

To run the binary on an input tflite  file:

    ./bazel-bin/xcore-opt <input_file.tflite> -o <output_file.tflite>

To view the various supported options:

    ./bazel-bin/xcore-opt --help

Test instructions
--
We use [llvm-lit](https://llvm.org/docs/CommandGuide/lit.html) for writing unit tests. This makes it easier to test the impact of a single pass. To run all lit tests in the `Test` folder, use the following command:

    bazel test //Test:all

An individual test can be run by combining the `test filename` and `.test` suffix to create the bazel target name. For example, the following command:

    bazel test //Test:xcfc-to-tflcustom.mlir.test
