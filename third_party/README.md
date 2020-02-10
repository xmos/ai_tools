Flatbuffers
-----------

You can experiment with tflite model flatbuffer files using the JSON format.
The flatbuffer schema compiler (`flatc`) can be used to convert the binary format to 
JSON text.

If you are planning to use `flatc` to convert `.tflite` models to/from JSON, be aware that it may lose precision on floating point numbers (see https://github.com/google/flatbuffers/issues/5371).
In particular, the python implementation of the graph transformer relies heavily on `flatc`.

The repo includes binaries under 'flatbuffers_xmos/flatc_{linux,darwin}' that fix the above issue, so you can use the graph transformer tool without having to build from source.
The tools in this repo should use these as defaults for the `flatc` binary.
Note that if you want to use another version (e.g. from your venv or system level install) you will have to specify that.

If you want to build from source, start by cloning the fork https://github.com/lkindrat-xmos/flatbuffers.
Then you can build using:

    > mkdir build
    > cd build/
    > ccmake ../
    > cmake ../
    > make
    > sudo make install

To convert a `.tflite` model flatbuffer file to JSON:

    > flatc --json path/to/tensorflow/tensorflow/lite/schema/schema.fbs -- path/to/your/model.tflite

To convert a JSON text file to `.tflite` model flatbuffer file:

    > flatc --binary path/to/tensorflow/tensorflow/lite/schema/schema.fbs path/to/your/model.json

See https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html for more information on `flatc`.
