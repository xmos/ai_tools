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
    sh "cmake --build . -t install --parallel 8 --config Release"
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
      } else if (platform == "mac") {
        unstash "mac_wheel"
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
    } else if (platform == "linux" | platform == "mac") {
      body()
    }
    junit "**/*_junit.xml"
  }
}

pipeline {
  agent none
  environment {
    REPO = "ai_tools"
    BAZEL_CACHE_URL = 'http://srv-bri-bld-cache.xmos.local:8080'
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
  stages { stage("On PR") { 
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
            dir("python/xmos_ai_tools/runtime") {
              unstash "release_archive"
              sh "unzip release_archive.zip lib/libxtflitemicro.a -d ./"
            }
            script {
              USER_ID = sh(script: 'id -u', returnStdout: true).trim()
              GROUP_ID = sh(script: 'id -g', returnStdout: true).trim()
              withEnv(['USER='+USER_ID, "XDG_CACHE_HOME=${env.WORKSPACE}/.cache", "TEST_TMPDIR=${env.WORKSPACE}/.cache", "TMPDIR=${env.WORKSPACE}/.cache"]) {
                docker.image('tensorflow/build:2.15-python3.10').inside("-e SETUP_SCM_PRETEND_VERSION=${env.TAG_VERSION} -u \"${USER_ID}:${GROUP_ID}\"") {
                  // get latest pip
                  sh "pip uninstall pip --yes"
                  sh "wget https://bootstrap.pypa.io/get-pip.py"
                  sh "python get-pip.py"
                  // install cmake
                  sh "pip install cmake"
                  // build host lib
                  sh "CC=/dt9/usr/bin/gcc CXX=/dt9/usr/bin/g++ ./build.sh -T xinterpreter-nozip -b"
                  dir("xformer") {
                    sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64"
                    sh "chmod +x bazelisk-linux-amd64"
                    sh """
                      ./bazelisk-linux-amd64 build //:xcore-opt \\
                        --verbose_failures \\
                        --linkopt=-lrt \\
                        --crosstool_top="@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain" \\
                        --//:disable_version_check \\
                        --crosstool_top="@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain" \\
                        --jobs 8
                    """
                  }
                  dir("python") {
                    sh "python setup.py bdist_wheel"
                  }
                }
              }
              withVenv { dir("python") {
                sh "pip install patchelf auditwheel==5.2.0 --no-cache-dir"
                sh "auditwheel repair --plat manylinux2014_x86_64 dist/*.whl"
                stash name: "linux_wheel", includes: "dist/*"
              } }
            }
          } } 
          stage("Build Mac runtime") {
            agent { label "macos && arm64 && xcode" }
            steps {
              setupRepo()
              createZip("mac_arm")
              extractRuntime()
              buildXinterpreter()
              // TODO: Fix this, use a rule for the fat binary instead of manually combining
              dir("xformer") {
                sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64"
                sh "chmod +x bazelisk-darwin-arm64"
                script {
                  def compileAndRename = { arch ->
                    def cpuFlag = arch == 'arm64' ? 'darwin_arm64' : 'darwin_x86_64'
                    def outputName = "xcore-opt-${arch}"
                    sh """
                      ./bazelisk-darwin-arm64 build //:xcore-opt \\
                        --cpu=${cpuFlag} \\
                        --copt=-fvisibility=hidden \\
                        --copt=-mmacosx-version-min=10.15 \\
                        --linkopt=-mmacosx-version-min=10.15 \\
                        --linkopt=-dead_strip \\
                        --//:disable_version_check
                      mv bazel-bin/xcore-opt ${outputName}
                    """
                  }
                  compileAndRename('arm64')
                  compileAndRename('x86_64')
                }
                sh "lipo -create xcore-opt-arm64 xcore-opt-x86_64 -output bazel-bin/xcore-opt"
              }
              createVenv("requirements.txt")
              dir("python") { withVenv {
                sh "pip install wheel setuptools setuptools-scm numpy six --no-cache-dir"
                sh "python setup.py bdist_wheel --plat macosx_10_15_universal2"
                stash name: "mac_wheel", includes: "dist/*"
              } }
            }
            post { cleanup { xcoreCleanSandbox() } }
          }
        }
    }
    stage("Test") { parallel {
      stage("Linux Test") { steps { script {
        withVenv { dir("xformer") {
          sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64"
          sh "chmod +x bazelisk-linux-amd64"
          sh "./bazelisk-linux-amd64 --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
        } }
        runTests("linux", dailyHostTest)
        withVenv {
          sh "pip install pytest nbmake"
          sh "pytest --nbmake ./docs/notebooks/*.ipynb"
        }
      } } } 
      stage("Mac arm64 Test") {
        agent { label "macos && arm64 && !macos_10_14" }
        steps { script {
          runTests("mac", dailyHostTest)
        } }
        post { cleanup {xcoreCleanSandbox() } }
      }
      stage("Mac x86_64 Test") {
        agent { label "macos && x86_64 && !macos_10_14" }
        steps { script {
          runTests("mac", dailyHostTest)
        } }
        post { cleanup {xcoreCleanSandbox() } }
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
      } }
      // stage("Publish") { steps {
      //   script {
      //     // if (params.TAG_VERSION != "") {
      //     dir("python") {
      //       unstash "linux_wheel"
      //       unstash "mac_wheel"
      //       withVenv {
      //         sh "pip install twine"
      //         sh "twine upload dist/*"
      //       }
      //     }
      //     // }
      //   }
      // } }
      }
    }
    post { cleanup { xcoreCleanSandbox() } }
  } }
}
