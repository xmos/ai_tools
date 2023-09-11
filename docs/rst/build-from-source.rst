Compiling xmos-ai-tools from source
======================

Requirements:
#############

* Install version 15 or later of the `XMOS tools <https://www.xmos.ai/software-tools/>`_ from your preferred location and activate it by sourcing ``SetEnv`` in the installation root.

* `CMake 3.23 <https://cmake.org/download/>`_ or newer is required for building libraries and test firmware.

* Clone the repository::

    git clone git@github.com:xmos/ai_tools.git
    cd ai_tools

* Set up and activate a Python virtual environment (or conda)::

    python -m venv ./venv
    . ./venv/bin/activate 

* Install the necessary python packages using ``pip``  inside the environment::

    pip install -r ./requirements.txt

Building:
#########

* Clone submodules and apply patch to ``tflite-micro``::

    ./build.sh -T init

* Build ``xformer`` and ``xinterpreter`` with the default settings::

    ./build.sh -T all -b

* (Optional) other ``./build.sh`` flags:

    * ``-d`` (debug) flag for a debug build.
    * ``-c`` (clean) to remove build artifacts.
    * ``-t`` (test) to run integration tests (on host)
    * ``-l`` (lsp) to generate ``compile_commands.json``

* Install the Python package in your environment::

    cd python && pip install .
