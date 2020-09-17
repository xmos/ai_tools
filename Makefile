NUM_PROCS := 4

.DEFAULT_GOAL := all

.PHONY: lib_nn_test_build
lib_nn_test_build:
	cd lib_nn/test/unit_test && make clean
	cd lib_nn/test/unit_test && make all

.PHONY: lib_nn_test_run
lib_nn_test_run: lib_nn_test_build
	cd lib_nn/test/unit_test && make run

.PHONY: utils_test_build
utils_test_build:
	cd utils && ./build.sh

.PHONY: tflite2xcore_test
tflite2xcore_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS)

.PHONY: integration_test
integration_test:
	cd utils/model_generation && pytest integration_test --cache-clear --collect-only -qq
	cd utils/model_generation && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

.PHONY: ci 
ci: lib_nn_test_build tflite2xcore_test integration_test

.PHONY: all
all: lib_nn_test_build tflite2xcore_test lib_nn_test_run integration_test
