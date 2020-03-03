# Integration Tests

Before running the integration tests, you must first build the `test_model` example application.

To build the `test_model` example application, follow the instructions located in:

    examples/apps/test_model

By default, all tests are configured to run using the xCORE simulator.  You must have the 
xCORE toolchain installed and in your system path.  

To run all tests

    > pytest

To run tests using multiple CPUs

    > pytest -n NUM

To run a single test

    > pytest -k name_of_test

To run the tests on the host PC

    > pytest --test-app=../examples/apps/test_model/bin/test_model
