@Library('xmos_jenkins_shared_library@v0.14.2') _

getApproval()

pipeline {
    agent {
        dockerfile {
            args "-v /home/jenkins/.keras:/root/.keras -v /etc/passwd:/etc/passwd:ro"
        }
    }

    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.0.6',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        booleanParam( // use to check results of rolling all conda deps forward
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
        )
    }

    options { // plenty of things could go here
        //buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage("Setup") {
            // Clone and install build dependencies
            steps {
                // clean auto default checkout
                sh "rm -rf *"
                // clone
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [[$class: 'SubmoduleOption',
                                  threads: 8,
                                  timeout: 20,
                                  shallow: true,
                                  parentCredentials: true,
                                  recursiveSubmodules: true],
                                 [$class: 'CleanCheckout']],
                    userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                         url: 'git@github.com:xmos/ai_tools']]
                ])
                // create venv and install pip packages
                sh "conda env create -q -p ai_tools_venv -f ./utils/environment.yml"
                sh """. activate ./ai_tools_venv &&
                      pip install -e "./tflite2xcore[test,examples]"
                """
                // Install xmos tools version
                sh "/XMOS/get_tools.py " + params.TOOLS_VERSION
            }
        }
        stage("Update all packages") {
            // Roll all conda packages forward beyond their pinned versions
            when { expression { return params.UPDATE_ALL } }
            steps {
                sh "conda update --all -y -q -p ai_tools_venv"
            }
        }
        stage("Build") {
            steps {
                // below is how we can activate the tools, NOTE: xTIMEcomposer -> XTC at tools 15.0.5 and later
                // sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv && //
                sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv &&
                      . activate ./ai_tools_venv &&
                      make clean &&
                      make build
                """
                sh ". activate ./ai_tools_venv && make tflite2xcore_dist"
                sh """. activate ./ai_tools_venv && cd experimental/xformer &&
                      bazel build --remote_cache=http://srv-bri-bld-cache:8080 //:xcore-opt --verbose_failures
                """
                sh """. activate ./ai_tools_venv &&
                      pip install -e "./third_party/lib_tflite_micro/tflm_interpreter[test]"
                """
            }
        }
        stage("Test") {
            steps {
                //sh """. activate ./ai_tools_venv &&
                      //make test NUM_PROCS=\$(grep -c ^processor /proc/cpuinfo)
                //"""
                // Any call to pytest can be given the "--junitxml SOMETHING_junit.xml" option
                // This step collects these files for display in Jenkins UI
                junit "**/*_junit.xml"
                //sh """. activate ./ai_tools_venv &&
                      //make integration_test NUM_PROCS=\$(grep -c ^processor /proc/cpuinfo)
                //"""
                // xformer2 integration tests
                sh """. activate ./ai_tools_venv &&
                      make xformer2_integration_test NUM_PROCS=\$(grep -c ^processor /proc/cpuinfo)
                """
            }
        }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
