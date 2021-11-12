NUM_PROCS := 4

.DEFAULT_GOAL := help

#**************************
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd third_party/lib_tflite_micro/ && make build

#**************************
# tflite2xcore targets
#**************************

.PHONY: lib_flexbuffers_build
lib_flexbuffers_build:
	cd utils/lib_flexbuffers && bash build.sh

.PHONY: tflite2xcore_unit_test
tflite2xcore_unit_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS) --junit

.PHONY: tflite2xcore_dist
tflite2xcore_dist:
	cd tflite2xcore && bash build_dist.sh

#**************************
# integration test targets
#**************************

.PHONY: integration_test
integration_test:
	cd test && pytest integration_test --cache-clear --collect-only -qq
	cd test && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

.PHONY: xformer2_test
xformer2_integration_test:
	cd test && pytest integration_test --cache-clear --collect-only -qq
	# conv2d tests
	cd test && pytest integration_test/test_single_op_models/test_conv2d --only-experimental-xformer2 -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

#**************************
# default build and test targets
#**************************

.PHONY: build
build: lib_flexbuffers_build xcore_interpreters_build

.PHONY: test
test: tflite2xcore_unit_test integration_test

#**************************
# other targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: clean
clean:
	cd third_party/lib_tflite_micro/ && make clean
	rm -rf utils/lib_flexbuffers/build

.PHONY: help
help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   build                         Build all components)
	$(info   test                          Run all tests (requires tflite2xcore[test] package))
	$(info   clean                         Clean all build artifacts)
	$(info )
	$(info secondary targets:)
	$(info   lib_flexbuffers_build         Build lib_flexbuffers)
	$(info   xcore_interpreters_build      Build xcore_interpreters)
	$(info   tflite2xcore_unit_test        Run tflite2xcore unit tests (requires tflite2xcore[test] package))
	$(info   tflite2xcore_dist             Build tflite2xcore distribution (requires tflite2xcore[test] package))
	$(info   integration_test              Run integration tests (requires tflite2xcore[test] package))
	$(info   xformer2_test                 Run integration tests with xformer2 (experimental requires tflite2xcore[test] package))
	$(info )

.PHONY: init_linux
init_linux:
	git submodule update --depth=1 --init --recursive --jobs 8
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO "https://github.com/bazelbuild/bazel/releases/download/$${BAZEL_VERSION}/bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
	chmod +x bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh && \
	./bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh --prefix=$$PWD/bazel

.PHONY: init_darwin
init_darwin:
	git submodule update --depth=1 --init --recursive --jobs 8
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO "https://github.com/bazelbuild/bazel/releases/download/$${BAZEL_VERSION}/bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh" && \
	chmod +x bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh && \
	./bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh --prefix=$$PWD/bazel

.PHONY: init_windows
init_windows:
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO 'https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-windows-x86_64.exe'
	mv bazel-${BAZEL_VERSION}-windows-x86_64.exe bazel.exe

.PHONY: build_release_linux
build_release_linux:
	python3 -m venv .venv
	(. .venv/bin/activate && pip install -r requirements.txt)
	(. .venv/bin/activate && cd experimental/xformer && ../../bazel/bin/bazel build --config=linux_config //:xcore-opt --verbose_failures)
	mkdir -p Install/Linux/External/xformer
	cp bazel-bin/xcore-opt Install/Linux/External/xformer

.PHONY: build_release_darwin
build_release_darwin:
	python3 -m venv .venv
	(. .venv/bin/activate && pip install -r requirements.txt)
	(. .venv/bin/activate && cd experimental/xformer && ../../bazel/bin/bazel build --config=darwin_config //:xcore-opt --verbose_failures)
	mkdir -p Install/Linux/External/xformer
	cp bazel-bin/xcore-opt Install/Mac/External/xformer

.PHONY: build_release_windows
build_release_windows:
	python3 -m venv .venv
	(. .venv/bin/activate && pip install -r requirements.txt)
	(. .venv/bin/activate && cd experimental/xformer && ../../bazel build --config=windows_config //:xcore-opt --verbose_failures)
	mkdir -p Install/Linux/External/xformer
	cp bazel-bin/xcore-opt Install/Windows/External/xformer
