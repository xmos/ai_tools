NUM_PROCS := 8
.DEFAULT_GOAL := help
.PHONY: xcore_interpreters_build xinterpreters_smoke_test_host xformer2_test version_check build test submodule_update clean help patch xformer_build create_zip

xformer2_integration_test:
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n $(NUM_PROCS) --junitxml=integration_tests/integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n $(NUM_PROCS) --junitxml=integration_tests/integration_bnns_junit.xml

version_check:
	cd ./xformer && ./version_check.sh

submodule_update: 
	git submodule update --init --recursive --jobs $(NUM_PROCS)

clean:
	$(MAKE) -C python/xmos_ai_tools/xinterpreters/host/ clean

patch:
	$(MAKE) -C third_party/lib_tflite_micro patch

lsp_setup:
	cd xformer && bazel run @hedron_compile_commands//:refresh_all

# ARM options, debug option
xformer_build:
	cd xformer && bazel build //:xcore-opt

create_zip:
	cd third_party/lib_tflite_micro && mkdir -p build && cd build && cmake .. --toolchain=../lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake && $(MAKE) create_zip
	mv third_party/lib_tflite_micro/build/release_archive.zip python/xmos_ai_tools/runtime/release_archive.zip
	cd python/xmos_ai_tools/runtime && rm -rf lib include && unzip release_archive.zip && rm release_archive.zip

xcore_interpreters_build: create_zip
	$(MAKE) -C python/xmos_ai_tools/xinterpreters/host/ install

init: submodule_update patch

build: version_check xcore_interpreters_build xformer_build

test: xformer2_integration_test

help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   build                         Build all components)
	$(info   test                          Run all tests)
	$(info   clean                         Clean all build artifacts)
	$(info )
	$(info secondary targets:)
	$(info   xcore_interpreters_build      Build xcore_interpreters)
	$(info   xformer2_integration_test     Run integration tests with xformer2)
	$(info )
