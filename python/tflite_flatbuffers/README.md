These instructions are based on this thread: https://github.com/tensorflow/tensorflow/issues/16561

Clone flatbuffers

    > git clone https://github.com/google/flatbuffers

Build flatbuffers

    > mkdir build
    > cd build/
    > ccmake ../
    > cmake ../
    > make
    > sudo make install

Generate Python wrappers

    > flatc --python ../../../tensorflow/tensorflow/lite/schema/schema.fbs
    > flatc --python ../../../tensorflow/tensorflow/lite/schema/metadata_schema.fbs
