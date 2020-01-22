# Integration Tests

Before running the integration tests, you must first build the `test_model` example application
and generate the test data.

To build the `test_model` example application, follow the instructions located in:

    examples\test_model

To generate the integration test data run

    > ./generate_test_data.py

Note, generate_test_data.py takes a few minutes to complete.

By default, all tests are configured to run using the xCORE simulator.  You must have the 
xCORE toolchain installed and in your system path.  

To run all tests

    > pytest

To run a single test

    > pytest -k name_of_test

To run the tests on the host PC

    > pytest --test-model=../examples/test_model/bin/test_model
