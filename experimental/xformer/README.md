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