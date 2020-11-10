NUM_PROCS := 4

.DEFAULT_GOAL := help

#**************************
# lib_nn targets
#**************************

.PHONY: lib_nn_build
lib_nn_build:
	cd lib_nn/test/unit_test && make clean
	cd lib_nn/test/unit_test && make all

.PHONY: lib_nn_test
lib_nn_test: lib_nn_build
	cd lib_nn/test/unit_test && xrun --io bin/xcore/unit_test.xe

#**************************
# integration test targets
#**************************

.PHONY: utils_build
utils_build:
	cd utils && ./build.sh

.PHONY: integration_test
integration_test:
	cd utils/model_generation && pytest integration_test --cache-clear --collect-only -qq
	cd utils/model_generation && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

#**************************
# tflite2xcore targets
#**************************

.PHONY: tflite2xcore_unit_test
tflite2xcore_unit_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS) --junit

.PHONY: tflite2xcore_dist
tflite2xcore_dist:
	cd tflite2xcore && ./build_dist.sh

.PHONY: tflite2xcore_dist_test
tflite2xcore_dist_test:
	cd tflite2xcore && ./test_dist.sh

#**************************
# ci target
#**************************

.PHONY: ci 
#TODO: Add lib_nn_test target when CI system connected HW
ci: lib_nn_build utils_build tflite2xcore_unit_test tflite2xcore_dist_test integration_test

#**************************
# ALL tests target
#**************************

.PHONY: test
test: lib_nn_test tflite2xcore_unit_test utils_build integration_test

#**************************
# development targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: develop
develop: submodule_update utils_build

.PHONY: help
help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info targets:)
	$(info )
	$(info develop                 Update submodules and build utils)
	$(info ci                      Run continuous integration build and test (requires Conda environment))
	$(info test                    Run all tests (requires Conda environment & connected hardware))
	$(info integration_test        Run integration tests (requires Conda environment))
	$(info tflite2xcore_unit_test  Run tflite2xcore unit tests (requires Conda environment))
	$(info tflite2xcore_dist       Build tflite2xcore distribution (requires Conda environment))
	$(info tflite2xcore_dist_test  Run tflite2xcore distribution tests (requires Conda environment))
	$(info lib_nn_test             Run lib_nn tests)
	$(info )
