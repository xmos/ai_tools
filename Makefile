NUM_PROCS := 1

.DEFAULT_GOAL := all

.PHONY: lib_nn_test
lib_nn_test:
	cd lib_nn/test/nn_operators && make all
	cd lib_nn/test/nn_operators && make run

.PHONY: lib_nn_test_clean
lib_nn_test_clean:
	cd lib_nn/test/nn_operators && make clean

.PHONY: testing_utils
testing_utils: lib_nn
	cd utils && ./build.sh

.PHONY: tflite2xcore_test
tflite2xcore_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS)

.PHONY: integration_test
integration_test:
	cd tests && ./generate_test_data.py -n $(NUM_PROCS)
	cd tests && pytest -v -n $(NUM_PROCS)

.PHONY: clean
clean: lib_nn_test_clean

.PHONY: test 
test: lib_nn_test tflite2xcore_test integration_test

.PHONY: all
all: lib_nn testing_utils test

.PHONY: foobar
foobar:
	cd lib_nn/test/unit_test && make all
	cd lib_nn/test/unit_test && make run
	cd utils && ./build.sh
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n 1
