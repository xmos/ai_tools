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
    withTools(params.TOOLS_VX4_VERSION) {sh 'make build_vx4'}
    withTools(params.TOOLS_VERSION) {sh 'make build_xs3'}
    // Stage headers and package both device libraries using the XS3 toolchain.
    withTools(params.TOOLS_VERSION) {sh 'make build_install'}
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

def dailyDeviceTest = { ->
  sh 'pytest integration_tests/test_runner.py -k daily_device --device -n 1 --junitxml=integration_tests/integration_device_junit.xml'
}

def dailyHostTest = { ->
  sh 'pytest integration_tests/test_runner.py -k daily_host -n auto --junitxml=integration_tests/integration_host_junit.xml'
}

def runTests(String platform, Closure body) {
  setupRepo()
  createVenv(reqFile:'requirements.txt')
  withVenv {
    sh_bat 'pip install -r integration_tests/requirements.txt'
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
    if (platform == 'device') {
      sh "cd ${WORKSPACE} && git clone https://github0.xmos.com/xmos-int/xtagctl.git"
      sh "pip install -e ${WORKSPACE}/xtagctl"
      withTools(params.TOOLS_VERSION) {
        body()
      }
    } else if (platform == 'linux' | platform == 'mac' | platform == 'windows') {
      body()
    }
    junit '**/*_junit.xml'
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
    dir('examples') {
      xcoreBuild()
    }
  }
}

pipeline {
  agent none
  environment {
    REPO = 'ai_tools'
    BAZEL_USER_ROOT = "${WORKSPACE}/.bazel/"
    SETUPTOOLS_SCM_PRETEND_VERSION = "1.4.3.dev40"
  }

  parameters {
    string(
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
                        bat "for /f %%i in ('python -m setuptools_scm -c ..\\python\\pyproject.toml') do bazelisk-windows-amd64.exe --output_user_root c:\\jenkins\\_bzl build //:xcore-opt --config=ci_windows --action_env PYTHON_BIN_PATH=\"${PYTHON_BIN_PATH}\" --action_env BAZEL_VC=\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\" --define SETUPTOOLS_SCM_VERSION=%%i"
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

            stage('Linux Test') {
              steps { script {
                runTests('linux', dailyHostTest)
                withVenv {
                sh 'pip install pytest nbmake'
                sh 'pytest --nbmake ./docs/notebooks/*.ipynb'
              }}}
            } // stage('Linux Test')

            stage('Mac arm64 Test') {
              agent { label 'macos && arm64 && !macos_10_14' }
              steps { script {runTests('mac', dailyHostTest)}}
              post { cleanup { xcoreCleanSandbox() } }
            } // stage('Mac arm64 Test')

            stage('Windows Test') {
              agent { label 'ai && windows10' }
              steps { script {runTests('windows', dailyHostTest)}}
              post { cleanup { xcoreCleanSandbox() } }
            } // stage('Windows Test')

            stage('Device Test') {
              agent {label 'xcore.ai-explorer && lpddr && !macos'}
              steps {script {dir('sandbox/ai_tools') {runTests('device', dailyDeviceTest)}}}
              post {
                always {
                  archiveArtifacts artifacts: 'sandbox/ai_tools/examples/app_mobilenetv2/arena_sizes.csv', allowEmptyArchive: true
                }
                cleanup {
                  xcoreCleanSandbox()
                }
              }
            } // stage('Device Test')

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
                  withCredentials([usernamePassword(
                    credentialsId: '__CREDID__', 
                    usernameVariable: 'TWINE_USERNAME', 
                    passwordVariable: 'TWINE_PASSWORD')]) 
                  {
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
