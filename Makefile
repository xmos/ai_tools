NUM_PROCS := 4
CLOBBER_FLAG := '-c'

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
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd xcore_interpreters/python_bindings && bash build.sh $(CLOBBER_FLAG)
	cd xcore_interpreters/xcore_firmware && bash build.sh $(CLOBBER_FLAG)

.PHONY: xcore_interpreters_unit_test
xcore_interpreters_unit_test:
	cd xcore_interpreters/xcore_interpreters && pytest -n $(NUM_PROCS) --junitxml=xcore_interpreters_junit.xml

.PHONY: xcore_interpreters_dist
xcore_interpreters_dist:
	cd xcore_interpreters && bash build_dist.sh

.PHONY: xcore_interpreters_dist_test
xcore_interpreters_dist_test:
	cd xcore_interpreters && bash test_dist.sh

#**************************
# tflite2xcore targets
#**************************

.PHONY: lib_flexbuffers_build
lib_flexbuffers_build:
	cd utils/lib_flexbuffers && bash build.sh $(CLOBBER_FLAG)

.PHONY: tflite2xcore_unit_test
tflite2xcore_unit_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS) --junit

.PHONY: tflite2xcore_dist
tflite2xcore_dist:
	cd tflite2xcore && bash build_dist.sh

.PHONY: tflite2xcore_dist_test
tflite2xcore_dist_test:
	cd tflite2xcore && bash test_dist.sh

#**************************
# integration test targets
#**************************

.PHONY: integration_test
integration_test:
	cd test/model_generation && pytest integration_test --cache-clear --collect-only -qq
	cd test/model_generation && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

#**************************
# ci target
#**************************

.PHONY: ci 
#TODO: Add lib_nn_test target when CI system connected HW
ci: CLOBBER_FLAG = '-c'
ci: lib_nn_build \
 lib_flexbuffers_build \
 tflite2xcore_unit_test \
 tflite2xcore_dist_test \
 xcore_interpreters_build \
 xcore_interpreters_unit_test \
 xcore_interpreters_dist_test \
 integration_test

#**************************
# ALL tests target
#**************************

.PHONY: test
test: lib_nn_test \
 lib_flexbuffers_build \
 tflite2xcore_unit_test \
 xcore_interpreters_unit_test \
 integration_test

#**************************
# development targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: _develop
_develop: submodule_update lib_flexbuffers_build xcore_interpreters_build

.PHONY: develop
develop: CLOBBER_FLAG=''
develop: _develop

.PHONY: clobber
clobber: CLOBBER_FLAG='-c'
clobber: _develop

.PHONY: help
help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   develop                       Update submodules and build xcore_interpreters)
	$(info   clobber                       Update submodules and build xcore_interpreters with clobber flag enabled)
	$(info   ci                            Run continuous integration build and test (requires Conda environment))
	$(info   integration_test              Run integration tests (requires Conda environment))
	$(info   test                          Run all tests (requires Conda environment & connected hardware))
	$(info )
	$(info lib_nn targets:)
	$(info   lib_nn_build                  Run lib_nn build)
	$(info   lib_nn_test                   Run lib_nn tests)
	$(info )
	$(info tflite2xcore targets:)
	$(info   lib_flexbuffers_build         Run lib_flexbuffers build)
	$(info   tflite2xcore_unit_test        Run tflite2xcore unit tests (requires Conda environment))
	$(info   tflite2xcore_dist             Build tflite2xcore distribution (requires Conda environment))
	$(info   tflite2xcore_dist_test        Run tflite2xcore distribution tests (requires Conda environment))
	$(info )
	$(info xcore_interpreter targets:)
	$(info   xcore_interpreters_build      Run xcore_interpreters build)
	$(info   xcore_interpreters_unit_test  Run xcore_interpreters unit tests (requires Conda environment))
	$(info   xcore_interpreters_dist       Build xcore_interpreters distribution (requires Conda environment))
	$(info   xcore_interpreters_dist_test  Run xcore_interpreters distribution tests (requires Conda environment))
	$(info )
