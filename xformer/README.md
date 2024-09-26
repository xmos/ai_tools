Build instructions
--

`Bazel` is used to build this program. The choice of build system is
driven by the most complex dependency (Tensorflow). 

Note before the first step ensure that you are in the conda venv as 
mentioned [here](https://github.com/xmos/ai_tools#readme) and 
have followed all steps including installing the necessary Python 
packages from `requirements.txt`.

Also note that submodules need to be cloned and tflite-micro needs to be patched before building `xformer`.
The following command should be run from this repo's root:

    ./build.sh -T init

See instructions [here](https://github.com/xmos/ai_tools/blob/develop/docs/rst/build-from-source.rst)

With Bazel installed (check `.bazelversion` for current version or use [bazelisk](https://github.com/bazelbuild/bazelisk) which handles this for you),
you can build with the following command (make sure you run it from the directory /ai_tools/xformer):

    bazel build //:xcore-opt

or if you are using bazelisk:

    bazelisk build //:xcore-opt

To run the binary on an input tflite  file:

    ./bazel-bin/xcore-opt <input_file.tflite> -o <output_file.tflite>

To view the various supported options:

    ./bazel-bin/xcore-opt --help


Python package instructions
--

After building the xcore-opt binary, the python package can be built
with the following command (make sure you run it from the directory 
/ai_tools/xformer/python):

    cd python
    python setup.py bdist_wheel

The wheel file will be created in a newly created `dist` folder.
This can now be installed via pip or uploaded to pypi via twine.    

Test instructions
--

We use [llvm-lit](https://llvm.org/docs/CommandGuide/lit.html) for writing unit tests. This makes it easier to test the impact of a single pass. To run all lit tests in the `Test` folder, use the following command:

    bazel test //Test:all

An individual test can be run by combining the `test filename` and `.test` suffix to create the bazel target name. For example, the following command:

    bazel test //Test:xcfc-to-tflcustom.mlir.test
