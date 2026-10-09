// This file relates to internal XMOS infrastructure and should be ignored by external users

@Library('xmos_jenkins_shared_library@v0.46.0') _

if (env.job_type != 'beta_release' && env.job_type != 'official_release') {
  getApproval()
}

def setupRepo() {
  println "Stage running on: ${env.NODE_NAME}"
  checkout scm
  sh 'git submodule update --init --recursive --jobs 8'
}

def doVersionCheck() {
  createVenv()
  withVenv {
    sh 'python -m pip install pytest'
    sh 'python -m pytest -q integration_tests/test_version_check.py'
  }
}

def createDeviceZip() {
  // build device runtime (vx4), (xs3), and install lib (xs3 is used)
  // Native build is a host-side compile check, not part of the device archive.
  dir('third_party/lib_tflite_micro') {  
    withTools(params.TOOLS_VX4_VERSION) {sh 'make build_vx4'}
    withTools(params.TOOLS_VERSION)     {sh 'make build_xs3'}
    withTools(params.TOOLS_VERSION)     {sh 'make build_install'}
    sh 'make build'
    stash name: 'release_archive', includes: 'build_xs3/release_archive.zip'
  }
}

def buildXinterpreterAndHostLib() {
  dir('python/xmos_ai_tools/xinterpreters') {
    sh 'cmake -B build'
    sh 'cmake --build build --target install --parallel 8 --config Release'
  }
}

def extractDeviceZipAndHeaders() {
  dir('python/xmos_ai_tools/runtime') {
    unstash 'release_archive'
    sh 'unzip -o build_xs3/release_archive.zip'
  }
}

def installWheel(String wheelStash) {
  dir('python') {
    unstash wheelStash
    sh 'pip install --force-reinstall dist/*'
  }
}

def runTestsHost(Map options) {
  setupRepo()
  createVenv(reqFile: 'integration_tests/requirements.txt')
  withVenv {
    installWheel(options.wheelStash)
    sh 'pytest integration_tests/test_runner.py -k daily_host -n auto --junitxml=integration_tests/integration_host_junit.xml'
    junit '**/*_junit.xml'
  }
}

def runTestsDevice(Map options) {
  setupRepo()
  createVenv(reqFile: 'integration_tests/requirements.txt')
  withVenv {
    installWheel(options.wheelStash)
    sh 'pip install git+https://github0.xmos.com/xmos-int/xtagctl.git'
    withTools(params.TOOLS_VERSION) {
      sh 'pytest integration_tests/test_runner.py -k daily_device --device -n 1 --junitxml=integration_tests/integration_device_junit.xml'
    }
    junit '**/*_junit.xml'
  }
}

def buildExamples() {
  setupRepo()
  createVenv(reqFile: 'requirements.txt')
  withVenv {
    installWheel('linux_wheel')
    dir('examples') {
      xcoreBuild()
    }
  }
}

def buildXformer(Map options) {
  def bazelBin = options.bazelBin
  def bazelConfig = options.bazelConfig
  def buildArgs = options.buildArgs ? " ${options.buildArgs}" : ''
  def commonArgs = "--config=${bazelConfig} --define SETUPTOOLS_SCM_VERSION=${env.SETUPTOOLS_SCM_PRETEND_VERSION}"

  dir('xformer') {
    sh "curl -fL ${env.BAZELISK_RELEASE_URL}/${bazelBin} -o ${bazelBin}"
    if (options.executable) {sh "chmod +x ${bazelBin}"}
    sh "./${bazelBin} build //:xcore-opt ${commonArgs} ${buildArgs}"
    if (options.runTests != false) {
      sh "./${bazelBin} test //Test:all ${commonArgs} ${buildArgs}"
    }
  }
}

def buildPyWheel(String platform) {
  def extra = (platform == 'mac') ? '--plat macosx_10_15_universal2' : ''
  dir('python') {
    sh "python setup.py bdist_wheel ${extra}"
    if (platform == 'linux') {
      sh 'auditwheel repair --plat manylinux_2_31_x86_64 dist/*.whl'
      sh 'rm dist/*.whl && mv wheelhouse/*.whl dist/'
    }
    stash name: "${platform}_wheel", includes: 'dist/*'
    archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
  }
}

def cleanXformer(String bazelBin) {
  dir('xformer') {
    sh "./${bazelBin} clean --expunge"
    sh "./${bazelBin} shutdown"
  }
}

pipeline {
  agent none
  environment {
    REPO = 'ai_tools'
    BAZELISK_RELEASE_URL = 'https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0'
    SETUPTOOLS_SCM_PRETEND_VERSION = "1.4.3.dev50"
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
            doVersionCheck()
            createVenv(reqFile: 'requirements.txt')
            withVenv { createDeviceZip() }
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
                    buildXformer(
                      bazelBin: 'bazelisk-linux-amd64',
                      bazelConfig: 'ci_linux',
                      executable: true
                    )
                    buildPyWheel('linux')
                  } // script
                } // withVenv
              } // steps
              post {
                cleanup {
                  cleanXformer('bazelisk-linux-amd64')
                  xcoreCleanSandbox()
                }
              }
            } // stage('Build linux runtime')

            stage('Build Windows runtime') {
              agent { label 'windows10 && ai' }
              steps {
                withVS() {
                  setupRepo()
                  extractDeviceZipAndHeaders()
                  buildXinterpreterAndHostLib()
                  createVenv(reqFile: 'python/requirements_build.txt')
                  withVenv {
                    buildXformer(
                      bazelBin: 'bazelisk-windows-amd64.exe',
                      bazelConfig: 'ci_windows',
                      runTests: false
                    )
                    buildPyWheel('windows')
                  }
                }
              }
              post { 
                cleanup {
                  cleanXformer('bazelisk-windows-amd64.exe')
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
                createVenv(reqFile: 'python/requirements_build.txt')
                withVenv {
                  script {
                    buildXformer(
                      bazelBin: 'bazelisk-darwin-arm64',
                      bazelConfig: 'ci_macos',
                      executable: true,
                      buildArgs: '--cpu=darwin_arm64'
                    )
                    dir('xformer') {
                      sh 'mv bazel-bin/xcore-opt xcore-opt-arm64'
                    }
                    buildXformer(
                      bazelBin: 'bazelisk-darwin-arm64',
                      bazelConfig: 'ci_macos',
                      buildArgs: '--cpu=darwin_x86_64'
                    )
                    dir('xformer') {
                      sh 'mv bazel-bin/xcore-opt xcore-opt-x86_64'
                      sh 'lipo -create xcore-opt-arm64 xcore-opt-x86_64 -output bazel-bin/xcore-opt'
                    }
                    buildPyWheel('mac')
                  } // script
                } // withVenv 
              } // steps
              post {
                cleanup {
                  cleanXformer('bazelisk-darwin-arm64')
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
                runTestsHost(wheelStash: 'linux_wheel')
                withVenv {
                sh 'pip install pytest nbmake'
                sh 'pytest --nbmake ./docs/notebooks/*.ipynb'
              }}}
            } // stage('Linux Test')

            stage('Mac arm64 Test') {
              agent { label 'macos && arm64 && !macos_10_14' }
              steps { script {runTestsHost(wheelStash: 'mac_wheel')}}
              post { cleanup { xcoreCleanSandbox() } }
            } // stage('Mac arm64 Test')

            stage('Windows Test') {
              agent { label 'ai && windows10' }
              steps { script {runTestsHost(wheelStash: 'windows_wheel')}}
              post { cleanup { xcoreCleanSandbox() } }
            } // stage('Windows Test')

            stage('Device Test') {
              agent {label 'xcore.ai-explorer && lpddr && !macos'}
              steps {script {dir('sandbox/ai_tools') {
                runTestsDevice(wheelStash: 'linux_wheel')
              }}}
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
                createVenv()
                withVenv {
                  withCredentials([usernamePassword(
                    credentialsId: 'PYPI_AITOOLS_TOKEN', 
                    usernameVariable: 'TWINE_USERNAME', 
                    passwordVariable: 'TWINE_PASSWORD')]) 
                  {
                    sh 'pip install twine'
                    sh 'twine upload --verbose dist/*'
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
