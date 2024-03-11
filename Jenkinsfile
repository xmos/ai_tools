// Things to optimise if this is too slow:
// - build device runtime in parallel with host runtimes, use mutex before combining into wheel

@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

def setupRepo() {
    script {
        println "Stage running on: ${env.NODE_NAME}"
        checkout scm
        sh "git submodule update --init --recursive --jobs 4"
        sh "make -C third_party/lib_tflite_micro patch"
    }
}

def createZip(String platform) {
    script {
        dir("xformer") { sh "./version_check.sh" }
        dir("third_party/lib_tflite_micro") {
            sh "mkdir -p build"
            dir("build") {
                if (platform == "device") {
                    sh "cmake .. --toolchain=../lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake"
                    sh "make create_zip -j4"
                } else {
                    sh "cmake -G 'Unix Makefiles' .. -DLIB_NAME=tflitemicro_${platform}"
                    sh "make create_zip -j4" 
                }
            }
        }
    }
}

def buildXinterpreter() {
    sh "mkdir -p python/xmos_ai_tools/xinterpreters/build"
    dir("python/xmos_ai_tools/xinterpreters/build") {
        sh "cmake .."
        sh "cmake --build . -t install --parallel 4 --config Release"
    }
}

def extractRuntime() {
    sh "mv third_party/lib_tflite_micro/build/release_archive.zip python/xmos_ai_tools/runtime"
    dir("python/xmos_ai_tools/runtime") {
        sh "unzip release_archive.zip"
        sh "rm release_archive.zip"
        unstash "release_archive"
        sh "unzip release_archive.zip lib/libxtflitemicro.a -d ./"
    }
}

def runPytestDevice(String test, String args, String junit) {
    timeout(time: 60, unit: 'MINUTES') {
        sh "xtagctl reset_all XCORE-AI-EXPLORER"
        sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --device --junitxml=integration_tests/integration_device_${junit}_junit.xml"
    }
}

def runPytestHost(String test, String args, String junit) {
    sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --junitxml=integration_tests/integration_host_${junit}_junit.xml"
}

def dailyDeviceTest = {
    timeout(time: 20, unit: 'MINUTES') {
        sh "xtagctl reset_all XCORE-AI-EXPLORER"
        sh "pytest examples/app_mobilenetv2"
    }
    runPytestDevice("8x8/test_slice", "-n 1 --tc 1", "slice_1")
    runPytestDevice("8x8/test_lstm", "-n 1 --tc 1", "lstm_1")
    runPytestDevice("8x8/test_lstm", "-n 1", "lstm_5")
    runPytestDevice("complex_models/8x8/test_cnn_classifier", "-n 1 --tc 1", "cnn_classifier_1")
    runPytestDevice("complex_models/8x8/test_cnn_classifier", "-n 1", "cnn_classifier_5")
    runPytestDevice("8x8/test_softmax", "-n 1 --device", "softmax_5")
    runPytestDevice("8x8/test_detection_postprocess", "-n 1", "detection_postprocess_5")
    runPytestDevice("16x8/", "-n 1", "16x8_5")
}

def dailyHostTest = {
    runPytestHost("float32", "-n 8 --tc 1", "float32_1")
    runPytestHost("16x8", "-n 8 --tc 5", "16x8_5")
    runPytestHost("complex_models/8x8", "-n 2 --tc 1", "complex_8x8_5")
    runPytestHost("complex_models/float32", "-n 1 --tc 1", "complex_float32_5")
    runPytestHost("8x8", "-n 8 --tc 1", "8x8_1")
    runPytestHost("8x8", "-n 8", "8x8_5")
    runPytestHost("8x8", "--compiled -n 8", "compiled_8x8")
    runPytestHost("bnns", "--bnn -n 8", "bnns")
    runPytestHost("bnns", "--bnn --compiled -n 8", "compiled_bnns")
}

def runTests(String platform, Closure body) {
    println "Stage running on: ${env.NODE_NAME}"
    checkout scm
    sh "./build.sh -T init"
    createVenv("requirements.txt")
    withVenv {
        sh "pip install -r requirements.txt"
        dir ("python") {
            if (platform == "linux" | platform == "device") {
                unstash "linux_wheel"
            } else if (platform == "mac_arm") {
                unstash "mac_arm_wheel"
            }
            sh "pip install dist/*"
        }
        script {
            XMOS_AITOOLSLIB_PATH = sh(script: "python -c \"import xmos_ai_tools.runtime as rt; import os; print(os.path.dirname(rt.__file__))\"", returnStdout: true).trim()
            env.XMOS_AITOOLSLIB_PATH = XMOS_AITOOLSLIB_PATH
        }
        if (platform == "device") {
            sh "cd ${WORKSPACE} && git clone https://github0.xmos.com/xmos-int/xtagctl.git"
            sh "pip install -e ${WORKSPACE}/xtagctl"
            withTools(params.TOOLS_VERSION) {
                body()
            }
        } else if (platform == "host") {
            body()
        }
        junit "**/*_junit.xml"
    }
}

pipeline {
    agent none
    environment {
        REPO = "ai_tools"
        BAZEL_CACHE_URL = 'http://srv-bri-bld-cache:8080'
        BAZEL_USER_ROOT = "${WORKSPACE}/.bazel/"
        REBOOT_XTAG = '1'
    }
    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.2.1',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        string( 
            name: 'TAG_VERSION',
            defaultValue: '',
            description: 'The release version, leave empty to not publish a release'
        )
    }

    options {
        timestamps()
        skipDefaultCheckout()
        buildDiscarder(logRotator(
            numToKeepStr:         env.BRANCH_NAME ==~ /develop/ ? '100' : '',
            artifactNumToKeepStr: env.BRANCH_NAME ==~ /develop/ ? '100' : ''
        ))
    }
    stages {
        stage("On PR") { 
            when { branch pattern: "PR-.*", comparator: "REGEXP" }
            agent { label "linux && x86_64 && !noAVX2" } 
            stages {
                stage("Build device runtime") { steps {
                    setupRepo()
                    createVenv("requirements.txt")
                    withVenv { sh "pip install -r requirements.txt" }
                    withVenv { withTools(params.TOOLS_VERSION) { createZip("device") } }
                    dir("third_party/lib_tflite_micro/build/") {
                        stash name: "release_archive", includes: "release_archive.zip"
                    }
                } }
                stage("Build host wheels") {
                    parallel {
                        stage("Build linux runtime") { steps {
                            withTools(params.TOOLS_VERSION) { createZip("linux") }
                            extractRuntime()
                            buildXinterpreter() 
                            // TODO: Docker!!!
                            // sh "mkdir -p .bazel-cache"
                            // script {
                            //     docker.image('tensorflow/build:2.14-python3.9').inside("-e SETUPTOOLS_SCM_PRETEND_VERSION=${env.TAG_VERSION} -v ${env.WORKSPACE}:/ai_tools -v .bazel-cache:/.cache -w /ai_tools") {
                            // sh "pip install auditwheel==5.2.0 cmake --no-cache-dir"
                            // sh "mkdir -p python/xmos_ai_tools/xinterpreters/build"
                            // dir("python/xmos_ai_tools/xinterpreters/build") {
                            //     sh "cmake .."
                            //     sh "cmake --build . -t install --parallel 4 --config Release"
                            // }
                            dir("xformer") {
                                sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64"
                                sh "chmod +x bazelisk-linux-amd64"
                                sh "./bazelisk-linux-amd64 build //:xcore-opt --verbose_failures --linkopt=-lrt  --//:disable_version_check --remote_cache=${env.BAZEL_CACHE_URL}"
                            }
                            // sh """
                            //     git config --global --add safe.directory /ai_tools
                            //     git config --global --add safe.directory /ai_tools/third_party/lib_nn
                            //     git config --global --add safe.directory /ai_tools/third_party/lib_tflite_micro
                            //     git config --global --add safe.directory /ai_tools/third_party/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro
                            //     git describe --tags
                            // """
                            withVenv { dir("python") {
                                sh "python setup.py bdist_wheel"
                            //     sh """
                            //         for f in dist/*.whl; do
                            //             auditwheel repair --plat manylinux2014_x86_64 $f
                            //         done
                            //     """
                                stash name: "linux_wheel", includes: "dist/*"
                            } }
                            //     } 
                            // }
                        } } 
                        stage("Build Arm Mac runtime") {
                            agent { label "macos && arm64 && xcode" }
                            steps {
                                setupRepo()
                                createZip("mac_arm")
                                extractRuntime()
                                buildXinterpreter()
                                dir("xformer") {
                                    sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64"
                                    sh "chmod +x bazelisk-darwin-arm64"
                                    sh "./bazelisk-darwin-arm64 build //:xcore-opt --cpu=darwin_arm64 --copt=-fvisibility=hidden --copt=-mmacosx-version-min=11.0 --linkopt=-mmacosx-version-min=11.0 --linkopt=-dead_strip --//:disable_version_check"
                                }
                                createVenv("requirements.txt")
                                dir("python") { withVenv {
                                    sh "pip install wheel setuptools setuptools-scm numpy six --no-cache-dir"
                                    sh "python setup.py bdist_wheel --plat-name macosx_11_0_arm64"
                                    stash name: "mac_arm_wheel", includes: "dist/*"
                                } }
                            }
                            post { cleanup { xcoreCleanSandbox() } }
                        }
                        stage("Build Windows runtime") {
                            agent { label "windows" }
                            steps { withVS() {
                                setupRepo()
                                createZip("windows")
                                extractRuntime()
                                buildXinterpreter()
                                withVenv { dir("xformer") {
                                    bat "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-windows-amd64.exe"
                                    bat "bazelisk-windows-amd64.exe --output_user_root c:\\_bzl build //:xcore-opt --action_env PYTHON_BIN_PATH='C:/hostedtoolcache/windows/Python/3.9.13/x64/python.exe' --//:disable_version_check --remote_cache=${env.BAZEL_CACHE_URL}"
                                }
                                createVenv("requirements.txt")
                                dir("python") { 
                                    bat "pip install wheel setuptools setuptools-scm numpy six --no-cache-dir"
                                    bat "python setup.py bdist_wheel"
                                    stash name: "windows_wheel", includes: "dist/*"
                                } }
                            } }
                            post { cleanup { xcoreCleanSandbox() } }
                        }
                    }
                }
                stage("Test") {
                    parallel {
                        stage("Linux Test") {
                            stage("Xformer Tests") { steps { withVenv {
                                sh "bazel --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
                            } } }
                            stage("Integration Tests") { steps { runTests("host", dailyHostTest) } }
                            stage("Notebook Tests") { steps { withVenv {
                                sh "pip install pytest nbmake"
                                sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                            } } }
                        }
                        stage("Device Test") {
                            agent { label "xcore.ai-explorer && lpddr && !macos" }
                            steps { script { runTests("device", dailyDeviceTest) } }
                            post {
                                always { 
                                    archiveArtifacts artifacts: 'examples/app_mobilenetv2/arena_sizes.csv', allowEmptyArchive: true
                                }
                                cleanup { xcoreCleanSandbox() }
                            }
                        }
                    }
                }
            }
            post { cleanup { xcoreCleanSandbox() } }
        } 
    }
}
