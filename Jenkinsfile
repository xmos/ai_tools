// This file relates to internal XMOS infrastructure and should be ignored by external users

@Library('xmos_jenkins_shared_library@v0.46.0') _

if (env.job_type != 'beta_release' && env.job_type != 'official_release') {
  getApproval()
}

def setupRepo() {
  println "Stage running on: ${env.NODE_NAME}"
  checkout scm
  sh 'git submodule update --init --recursive --jobs 4'
  sh 'make -C third_party/lib_tflite_micro patch'
}

def createDeviceZip() {
  dir('third_party/lib_tflite_micro') {
    // build device runtime (vx4), (xs3), and install lib (xs3 is used)
    withTools(params.TOOLS_VX4_VERSION) {sh 'make build_vx4'}
    withTools(params.TOOLS_VERSION)     {sh 'make build_xs3'}
    withTools(params.TOOLS_VERSION)     {sh 'make build_install'}
    // Native build is a host-side compile check, not part of the device archive.
    sh 'make build'
  }
}

def buildXinterpreterAndHostLib() {
  dir('python/xmos_ai_tools/xinterpreters') {
    sh 'cmake -S . -B build'
    sh 'cmake --build build --target install --parallel 8 --config Release'
  }
}

def extractDeviceZipAndHeaders() {
  dir('python/xmos_ai_tools/runtime') {
    unstash 'release_archive'
    sh 'unzip -o release_archive.zip'
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
    sh 'pip install -r integration_tests/requirements.txt'
    sh 'python -m pytest -q integration_tests/test_version_check.py'
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
        } // stage('Build device runtime')
        
        stage('Build host wheels') {
          parallel {

            stage('Build linux runtime') {
              steps {
                extractDeviceZipAndHeaders()
                buildXinterpreterAndHostLib()
                createVenv(reqFile: 'python/requirements_build.txt')
                withVenv {
                  script {
                    dir('xformer') {
                      sh 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64'
                      sh 'chmod +x bazelisk-linux-amd64'
                      sh './bazelisk-linux-amd64 build //:xcore-opt --config=ci_linux --define SETUPTOOLS_SCM_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}'
                      sh './bazelisk-linux-amd64 test //Test:all --config=ci_linux'
                    } // dir xformer
                    dir('python') {
                      sh 'python setup.py bdist_wheel'
                      sh 'pip install patchelf auditwheel --no-cache-dir'
                      sh 'auditwheel repair --plat manylinux_2_31_x86_64 dist/*.whl'
                      sh 'rm dist/*.whl && mv wheelhouse/*.whl dist/'
                      stash name: 'linux_wheel', includes: 'dist/*'
                      archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                    } // dir python
                  } // script
                } // withVenv
              } // steps
              post {
                unsuccessful { xcoreCleanSandbox() }
                cleanup {
                  dir('xformer') {
                    sh './bazelisk-linux-amd64 clean --expunge'
                  }
                }
              }
            } // stage('Build linux runtime')

            stage('Build Windows runtime') {
              agent { label 'ai && windows10' }
              steps {
                withVS() {
                  setupRepo()
                  extractDeviceZipAndHeaders()
                  buildXinterpreterAndHostLib()
                  createVenv(reqFile: 'python/requirements_build.txt')
                  withVenv {
                    dir('xformer') {
                      script {
                        sh 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-windows-amd64.exe'
                        sh 'bazelisk-windows-amd64.exe build //:xcore-opt --config=ci_windows --define SETUPTOOLS_SCM_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}'
                        sh 'bazelisk-windows-amd64.exe test //Test:all --config=ci_windows'
                      }
                    }
                    dir('python') {
                      script {
                        sh 'python setup.py bdist_wheel'
                      }
                      stash name: 'windows_wheel', includes: 'dist/*'
                      archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                    }
                  }
                }
              }
              post { 
                cleanup {
                  dir('xformer') {
                    sh 'bazelisk-windows-amd64.exe clean --expunge'
                    sh 'bazelisk-windows-amd64.exe shutdown'
                    script {
                      HANGING_BAZEL_EMBEDDED_JAVA_PID = bat(script: '@ps -W | grep _bzl | tr -s \" \" | cut -d \" \" -f 5', returnStdout: true).split()[0].trim()
                      sh "taskkill /F /PID \"${HANGING_BAZEL_EMBEDDED_JAVA_PID}\""
                    }
                  }
                  xcoreCleanSandbox() 
                } 
              }
            } // stage('Build Windows runtime')

            stage('Build Mac runtime') {
              agent { label 'macos && arm64 && xcode' }
              steps {
                setupRepo()
                extractDeviceZipAndHeaders()
                buildXinterpreterAndHostLib()
                // TODO: Fix this, use a rule for the fat binary instead of manually combining
                createVenv(reqFile: 'python/requirements_build.txt')
                withVenv {
                  script {
                    dir('xformer') { 
                        script {
                          sh 'curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64'
                          sh 'chmod +x bazelisk-darwin-arm64'
                          // mac arm64
                          sh './bazelisk-darwin-arm64 build //:xcore-opt --config=ci_macos --define SETUPTOOLS_SCM_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} --cpu=darwin_arm64'
                          sh 'mv bazel-bin/xcore-opt xcore-opt-arm64'
                          // mac intel
                          sh './bazelisk-darwin-arm64 build //:xcore-opt --config=ci_macos --define SETUPTOOLS_SCM_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} --cpu=darwin_x86_64'
                          sh 'mv bazel-bin/xcore-opt xcore-opt-x86_64'
                          // create fat binary
                          sh 'lipo -create xcore-opt-arm64 xcore-opt-x86_64 -output bazel-bin/xcore-opt'
                        }
                    } // dir('xformer')
                    dir('python') { 
                        script{
                          sh 'python setup.py bdist_wheel --plat macosx_10_15_universal2'
                        }
                        stash name: 'mac_wheel', includes: 'dist/*'
                        archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
                    } // dir('python')
                  } // script
                } // withVenv 
              } // steps
              post {
                cleanup {
                  xcoreCleanSandbox() 
                } // cleanup
              } // post
            } // stage('Build Mac runtime')
          } // Parallel
        } // Build host wheels

        stage('Build examples') {
          when {
            expression { env.job_type != 'beta_release' && env.job_type != 'official_release' }
          }
          steps {
            script { buildExamples() }
          }
          post { unsuccessful { xcoreCleanSandbox() } }
        } // stage('Build examples')

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
        } // stage('Test')

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
        } // stage('Publish')
      }
      post { cleanup { xcoreCleanSandbox() } }
  } }
}
