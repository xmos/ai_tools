// Things to optimise if this is too slow:
// -build device runtime in parallel with host runtimes, use mutex before combining into wheel

@Library('xmos_jenkins_shared_library@v0.46.0') _

if (env.job_type != 'beta_release' && env.job_type != 'official_release') {
  getApproval()
}

def sh_bat(cmd) {
  if (isUnix()) {
    sh cmd
  } else {
    bat cmd
  }
}

def setupRepo() {
  println "Stage running on: ${env.NODE_NAME}"
  checkout scm
  sh_bat 'git submodule update --init --recursive --jobs 4'
  sh_bat 'make -C third_party/lib_tflite_micro patch'
}

def createDeviceZip() {
  dir('third_party/lib_tflite_micro') {

    // build device runtime (vx4)
    withTools(params.TOOLS_VX4_VERSION) {
      sh 'make build_vx4'
    }
    withTools(params.TOOLS_VERSION) {
      sh 'make build_xs3'
    }
    // Stage headers and package both device libraries using the XS3 toolchain.
    withTools(params.TOOLS_VERSION) {
      sh 'make build_install'
    }
    // Native build is a host-side compile check, not part of the device archive.
    sh 'make build'
  }
}

def buildXinterpreterAndHostLib() {
  dir('python/xmos_ai_tools/xinterpreters') {
    sh_bat 'cmake -S . -B build'
    sh_bat 'cmake --build build --target install --parallel 8 --config Release'
  }
}

def extractDeviceZipAndHeaders() {
  dir('python/xmos_ai_tools/runtime') {
    unstash 'release_archive'
    sh_bat 'unzip -o release_archive.zip'
  }
}

def runPytestDevice(String test, String args, String junit) {
  timeout(time: 60, unit: 'MINUTES') {
    sh 'xtagctl reset_all XCORE-AI-EXPLORER'
    sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --device --junitxml=integration_tests/integration_device_${junit}_junit.xml"
  }
}

def runPytestHost(String test, String args, String junit) {
  sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --junitxml=integration_tests/integration_host_${junit}_junit.xml"
}

def dailyDeviceTest = {
  //TODO fix all jenkins infra to remove those
  // timeout(time: 20, unit: 'MINUTES') {
  //   sh 'xtagctl reset_all XCORE-AI-EXPLORER'
  //   sh 'pytest examples/app_mobilenetv2'
  // }
  runPytestDevice('8x8/test_broadcast', '-n 1 --tc 1', 'broadcast_1')
  runPytestDevice('16x8/test_transpose', '-n 1', '16x8_transpose')
  runPytestDevice('8x8/test_concatenate', '-n 1 --tc 5', 'concat_5')
  runPytestDevice('8x8/test_mean', '-n 1 --tc 1', 'mean_1')
  runPytestDevice('16x8/test_mean', '-n 1 --tc 1', '16x8_mean_1')
  runPytestDevice('8x8/test_lstm', '-n 1 --tc 1', 'lstm_1')
  runPytestDevice('8x8/test_lstm', '-n 1', 'lstm_5')
  runPytestDevice('complex_models/8x8/test_cnn_classifier', '-n 1 --tc 1', 'cnn_classifier_1')
  runPytestDevice('complex_models/8x8/test_cnn_classifier', '-n 1', 'cnn_classifier_5')
  runPytestDevice('8x8/test_softmax', '-n 1 --device', 'softmax_5')
  runPytestDevice('8x8/test_detection_postprocess', '-n 1', 'detection_postprocess_5')
  runPytestDevice('16x8/test_conv2d', '-n 1', '16x8_conv2d_5')
  runPytestDevice('16x8/test_transpose_conv', '-n 1', '16x8_transpose_conv_5')
}

def dailyHostTest = { platform ->
  runPytestHost('float32', '-n 8 --tc 1', 'float32_1')
  runPytestHost('16x8', '-n 8 --tc 5', '16x8_5')
  runPytestHost('complex_models/8x8', '-n 2 --tc 5', 'complex_8x8_5')
  runPytestHost('complex_models/float32', '-n 1 --tc 5', 'complex_float32_5')
  runPytestHost('8x8', '-n 8 --tc 1', '8x8_1')
  runPytestHost('8x8', '-n 8', '8x8_5')
  if (platform != 'windows') {
    // TODO - fix bnn tests on Windows
    runPytestHost('bnns', '--bnn -n 8', 'bnns')
    // TODO - fix compiled tests on Windows
    runPytestHost('8x8', '--compiled -n 8', 'compiled_8x8')
    runPytestHost('bnns', '--bnn --compiled -n 8', 'compiled_bnns')
    runPytestHost('complex_models/8x8/test_mobilenet_v2', '--compiled -n 8', 'compiled_mobilenetv2')
  }
}

def runTests(String platform, Closure body) {
  setupRepo()
  createVenv('requirements.txt')
  withVenv {
    sh 'pip install -r requirements.txt'
    sh 'pip install -r integration_tests/requirements.txt'
    sh_bat 'python -m pytest -q integration_tests/test_version_check.py'
    dir('python') {
      if (platform == 'linux' | platform == 'device') {
        unstash 'linux_wheel'
      } else if (platform == 'mac') {
        unstash 'mac_wheel'
      } else if (platform == 'windows') {
        unstash 'windows_wheel'
      }
      sh 'pip install dist/*'
    }
    script {
      XMOS_AITOOLSLIB_PATH = sh(script: 'python -c \"import xmos_ai_tools.runtime as rt; import os; print(os.path.dirname(rt.__file__))\"', returnStdout: true).trim()
      env.XMOS_AITOOLSLIB_PATH = XMOS_AITOOLSLIB_PATH
    }
    if (platform == 'device') {
      sh "cd ${WORKSPACE} && git clone https://github0.xmos.com/xmos-int/xtagctl.git"
      sh "pip install -e ${WORKSPACE}/xtagctl"
      withTools(params.TOOLS_VERSION) {
        body(platform)
      }
    } else if (platform == 'linux' | platform == 'mac' | platform == 'windows') {
      body(platform)
    }
    junit '**/*_junit.xml'
  }
}

def generateExampleSources() {
  dir('examples/app_no_flash') {
    sh 'xcore-opt vww_quant.tflite -o model.tflite'
    sh 'mv -f model.tflite.cpp model.tflite.h src/'
  }

  dir('examples/app_flash_single_model') {
    sh 'xcore-opt --xcore-weights-file=model.params vww_quant.tflite -o model.tflite'
    sh 'mv -f model.tflite.cpp model.tflite.h src/'
    sh '''python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
      output_file="xcore_flash_binary.out",
      model_files=["model.tflite"],
      param_files=["model.params"]
    )'
    '''
  }

  ['app_flash_two_models', 'app_flash_two_models_one_arena'].each { example ->
    dir("examples/${example}") {
      sh '''xcore-opt --xcore-weights-file=model1.params \
        --xcore-naming-prefix=model1_ \
        vww_quant1.tflite -o model1.tflite'''
      sh '''xcore-opt --xcore-weights-file=model2.params \
        --xcore-naming-prefix=model2_ \
        vww_quant2.tflite -o model2.tflite'''
      sh 'mv -f model1.tflite.cpp model1.tflite.h src/'
      sh 'mv -f model2.tflite.cpp model2.tflite.h src/'
      sh '''python -c 'from xmos_ai_tools import xformer as xf; xf.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=["model1.tflite", "model2.tflite"],
        param_files=["model1.params", "model2.params"]
      )'
      '''
    }
  }

  dir('examples/app_profiling') {
    sh 'xcore-opt vww_quant.tflite -o model.tflite'
    sh 'mv -f model.tflite.cpp model.tflite.h src/'
  }

  dir('examples/app_single_model_on_two_tiles') {
    sh 'python generate_optimized_cpp_for_xcore.py'
  }

  dir('examples/app_mobilenetv2') {
    sh 'python obtain_and_optimize_mobilenetv2.py'
  }

  dir('examples/app_audio_network') {
    sh 'xcore-opt denoise_16x8.tflite -o model_audioi16.tflite --xcore-thread-count=5'
    sh 'mv -f model_audioi16.tflite.cpp model_audioi16.tflite.h src/'
  }

  dir('examples/app_yolov8_classification') {
    sh 'python -m pip install -r requirements.txt'
    sh 'python obtain_and_optimize_yolov8_cls.py'
  }
}

def buildExamples() {
  setupRepo()
  createVenv(reqFile: 'requirements.txt')
  withVenv {
    dir('python') {
      unstash 'linux_wheel'
      sh 'python -m pip install dist/*'
    }
    sh 'rm -rf ../lib_xud'
    generateExampleSources()
    dir('examples') {
      xcoreBuild()
    }
  }
}

pipeline {
  agent none
  environment {
    REPO = 'ai_tools'
    BAZEL_CACHE_URL = 'http://srv-bri-bld-cache.xmos.local:8080'
    BAZEL_USER_ROOT = "${WORKSPACE}/.bazel/"
    SETUPTOOLS_SCM_PRETEND_VERSION = "1.4.3.dev40"
  }

  parameters { // Available to modify on the job page within Jenkins if starting a build
    string( // use to try different tools versions
      name: 'TOOLS_VERSION',
      defaultValue: '15.3.1',
      description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
    )
    string(
      name: 'TOOLS_VX4_VERSION',
      defaultValue: '-j --repo arch_vx_slipgate -b develop -a XTC 1184',
      description: 'The XTC Slipgate tools version'
    )
  }

  options {
    timestamps()
    skipDefaultCheckout()
    buildDiscarder(xmosDiscardBuildSettings())
  }

  stages { stage('On PR') {
      when { anyOf { branch pattern: 'PR-.*', comparator: 'REGEXP'; expression { env.job_type == 'beta_release' || env.job_type == 'official_release' } } }
      agent { label 'linux && x86_64 && !noAVX2' }
      stages {
        stage('Build device runtime') {
          steps {
            setupRepo()
            createVenv(reqFile: 'requirements.txt')
            withVenv { createDeviceZip() }
            dir('third_party/lib_tflite_micro/build_xs3/') {
              stash name: 'release_archive', includes: 'release_archive.zip'
            }
          }
          post {
            unsuccessful { xcoreCleanSandbox() }
          }
        }
        stage('Build host wheels') {
          parallel {
            stage('Build linux runtime') {
              steps {
                extractDeviceZipAndHeaders()
                script {
                  def customImage = docker.build("tensorflow-image-with-updated-pip:${env.BUILD_ID}")
                  USER_ID = sh(script: 'id -u', returnStdout: true).trim()
                  withEnv(['USER=' + USER_ID, "XDG_CACHE_HOME=${env.WORKSPACE}/.cache", "TEST_TMPDIR=${env.WORKSPACE}/.cache", "TMPDIR=${env.WORKSPACE}/.cache"]) {
                    customImage.inside() {
                      sh 'git describe --tags'
                      withEnv(['CC=/dt9/usr/bin/gcc', 'CXX=/dt9/usr/bin/g++']) {
                        buildXinterpreterAndHostLib()
                      }
                      dir('xformer') {
                        sh 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64'
                        sh 'chmod +x bazelisk-linux-amd64'
                        sh """
                        rm -rf /var/tmp/_bazel_jenkins/install/*
                        ./bazelisk-linux-amd64 build //:xcore-opt \\
                          --config=ci_linux \\
                          --crosstool_top="@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain" \\
                          --remote_cache=${env.BAZEL_CACHE_URL} \\
                          --define SETUPTOOLS_SCM_VERSION=\$(python -m setuptools_scm -c ../python/pyproject.toml)
                      """
                        sh '''
                        rm -rf /var/tmp/_bazel_jenkins/install/*
                        ./bazelisk-linux-amd64 test //Test:all \\
                          --config=ci_linux \\
                          --crosstool_top="@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain"
                      '''
                      }
                      dir('python') {
                        script {
                          if (env.job_type == 'official_release') {
                            withEnv(["SETUPTOOLS_SCM_PRETEND_VERSION=${env.TAG_VERSION}"]) {
                              sh 'python setup.py bdist_wheel'
                            }
                        } else {
                            sh 'python setup.py bdist_wheel'
                          }
                        }
                      }
                    }
                  }
                  withVenv { dir('python') {
                      sh 'pip install patchelf auditwheel==5.2.0 --no-cache-dir'
                      sh 'auditwheel repair --plat manylinux2014_x86_64 dist/*.whl'
                      sh 'rm dist/*.whl'
                      sh 'mv wheelhouse/*.whl dist/'
                      stash name: 'linux_wheel', includes: 'dist/*'
                      archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                } }
                }
              }
              post { unsuccessful { xcoreCleanSandbox() } }
            }
            stage('Build Windows runtime') {
              agent { label 'ai && windows10' }
              steps {
                withVS() {
                  setupRepo()
                  extractDeviceZipAndHeaders()
                  buildXinterpreterAndHostLib()
                  createVenv('requirements.txt')
                  withVenv {
                    bat 'pip install wheel setuptools setuptools-scm numpy six --no-cache-dir'
                    dir('xformer') {
                      bat 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-windows-amd64.exe'
                      script {
                        bat 'bazelisk-windows-amd64.exe clean --expunge'
                        PYTHON_BIN_PATH = bat(script: '@where python.exe', returnStdout: true).split()[0].trim()
                        bat "for /f %%i in ('python -m setuptools_scm -c ..\\python\\pyproject.toml') do bazelisk-windows-amd64.exe --output_user_root c:\\jenkins\\_bzl build //:xcore-opt --config=ci_windows --remote_cache=${env.BAZEL_CACHE_URL} --action_env PYTHON_BIN_PATH=\"${PYTHON_BIN_PATH}\" --action_env BAZEL_VC=\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\" --define SETUPTOOLS_SCM_VERSION=%%i"
                      }
                    }

                    dir('python') {
                      script {
                        if (env.job_type == 'official_release') {
                          withEnv(["SETUPTOOLS_SCM_PRETEND_VERSION=${env.TAG_VERSION}"]) {
                            bat 'python setup.py bdist_wheel'
                          }
                      } else {
                          bat 'python setup.py bdist_wheel'
                        }
                      }
                      stash name: 'windows_wheel', includes: 'dist/*'
                      archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                    }
                  }
                }
              }
              post { cleanup {
                  dir('xformer') {
                    bat 'bazelisk-windows-amd64.exe clean --expunge'
                    bat 'bazelisk-windows-amd64.exe shutdown'
                    script {
                      HANGING_BAZEL_EMBEDDED_JAVA_PID = bat(script: '@ps -W | grep _bzl | tr -s \" \" | cut -d \" \" -f 5', returnStdout: true).split()[0].trim()
                      bat "taskkill /F /PID \"${HANGING_BAZEL_EMBEDDED_JAVA_PID}\""
                    }
                  }
                  xcoreCleanSandbox() } }
            }
            stage('Build Mac runtime') {
              agent { label 'macos && arm64 && xcode' }
              steps {
                setupRepo()
                extractDeviceZipAndHeaders()
                buildXinterpreterAndHostLib()
                // TODO: Fix this, use a rule for the fat binary instead of manually combining
                createVenv('requirements.txt')
                dir('xformer') { withVenv {
                    sh 'pip install wheel setuptools setuptools-scm numpy six --no-cache-dir'
                    sh 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64'
                    sh 'chmod +x bazelisk-darwin-arm64'
                    script {
                      def compileAndRename = { arch ->
                        def cpuFlag = arch == 'arm64' ? 'darwin_arm64' : 'darwin_x86_64'
                        def outputName = "xcore-opt-${arch}"
                        sh """
                        rm -rf /var/tmp/_bazel_jenkins/install/*
                        ./bazelisk-darwin-arm64 build //:xcore-opt \\
                        --config=ci_macos \\
                        --cpu=${cpuFlag} \\
                        --remote_cache=${env.BAZEL_CACHE_URL} \\
                        --define SETUPTOOLS_SCM_VERSION=\$(python -m setuptools_scm -c ../python/pyproject.toml)
                      mv bazel-bin/xcore-opt ${outputName}
                    """
                      }
                      compileAndRename('arm64')
                      compileAndRename('x86_64')
                    }
                    sh 'lipo -create xcore-opt-arm64 xcore-opt-x86_64 -output bazel-bin/xcore-opt'
                } }
                dir('python') { withVenv {
                    script {
                      if (env.job_type == 'official_release') {
                        withEnv(["SETUPTOOLS_SCM_PRETEND_VERSION=${env.TAG_VERSION}"]) {
                          sh 'python setup.py bdist_wheel --plat macosx_10_15_universal2'
                        }
                    } else {
                        sh 'python setup.py bdist_wheel --plat macosx_10_15_universal2'
                      }
                    }
                    stash name: 'mac_wheel', includes: 'dist/*'
                    archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                } }
              }
              post { cleanup { xcoreCleanSandbox() } }
            }
          }
        }
        stage('Build examples') {
          when {
            expression { env.job_type != 'beta_release' && env.job_type != 'official_release' }
          }
          steps {
            script { buildExamples() }
          }
          post { unsuccessful { xcoreCleanSandbox() } }
        }
        stage('Test') {
          when {
            expression { env.job_type != 'beta_release' && env.job_type != 'official_release' }
          }
          parallel {
            stage('Linux Test') { steps { script {
                  runTests('linux', dailyHostTest)
                  withVenv {
                    sh 'pip install pytest nbmake'
                    sh 'pytest --nbmake ./docs/notebooks/*.ipynb'
                  }
      } } }
            stage('Mac arm64 Test') {
              agent { label 'macos && arm64 && !macos_10_14' }
              steps { script {
                  runTests('mac', dailyHostTest)
        } }
              post { cleanup { xcoreCleanSandbox() } }
            }
            // TODO Too old MacOS version
            // stage("Mac x86_64 Test") {
            //   agent { label "macos && x86_64" }
            //   steps { script {
            //     runTests("mac", dailyHostTest)
            //   } }
            //   post { cleanup {xcoreCleanSandbox() } }
            // }
            stage('Windows Test') {
              agent { label 'ai && windows10' }
              steps { script {
                  runTests('windows', dailyHostTest)
        } }
              post { cleanup { xcoreCleanSandbox() } }
            }
            stage('Device Test') {
              agent { label 'xcore.ai-explorer && lpddr && !macos' }
              steps { script { runTests('device', dailyDeviceTest) } }
              post {
                always {
                  archiveArtifacts artifacts: 'examples/app_mobilenetv2/arena_sizes.csv', allowEmptyArchive: true
                }
                cleanup { xcoreCleanSandbox() }
              }
            }
          }
        }
        stage('Publish') {
          when {
            expression { env.job_type == 'beta_release' || env.job_type == 'official_release' }
          }
          steps {
            script {
              dir('python') {
                unstash 'linux_wheel'
                unstash 'mac_wheel'
                unstash 'windows_wheel'
                archiveArtifacts artifacts: 'dist/*', allowEmptyArchive: true
                withVenv {
                  withCredentials([usernamePassword(credentialsId: '__CREDID__', usernameVariable: 'TWINE_USERNAME', passwordVariable: 'TWINE_PASSWORD')]) {
                    sh 'pip install twine'
                    sh 'twine upload dist/*'
                  }
                }
              }
            }
          }
        }
      }
      post { cleanup { xcoreCleanSandbox() } }
  } }
}
