NUM_PROCS := 4
CLOBBER_FLAG := '-c'

.DEFAULT_GOAL := help

#**************************
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd utils/ai_deployment_framework/xcore_interpreters/python_bindings && bash build.sh $(CLOBBER_FLAG)
	cd utils/ai_deployment_framework/xcore_interpreters/xcore_firmware && bash build.sh $(CLOBBER_FLAG)

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
	cd test && pytest integration_test --cache-clear --collect-only -qq
	cd test && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

#**************************
# ci target
#**************************

.PHONY: ci 
ci: CLOBBER_FLAG = '-c'
ci: lib_flexbuffers_build \
 tflite2xcore_unit_test \
 tflite2xcore_dist_test \
 xcore_interpreters_build \
 integration_test

#**************************
# ALL tests target
#**************************

.PHONY: test
test: lib_flexbuffers_build \
 tflite2xcore_unit_test \
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
	$(info secondary targets:)
	$(info   lib_flexbuffers_build         Run lib_flexbuffers build)
	$(info   tflite2xcore_unit_test        Run tflite2xcore unit tests (requires Conda environment))
	$(info   tflite2xcore_dist             Build tflite2xcore distribution (requires Conda environment))
	$(info   tflite2xcore_dist_test        Run tflite2xcore distribution tests (requires Conda environment))
	$(info )
