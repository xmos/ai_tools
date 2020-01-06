# Integration Tests

Before running the integration tests, follow the instructions for building the `test_model` 
example application located in:

    examples\test_model

By default, all tests are configured to run using the xCORE simulator.  You must have the 
xCORE toolchain installed and in your system path.  

To run all tests

    > pytest

To run the tests on the host PC

    > pytest --test-model=../examples/test_model/bin/test_model

