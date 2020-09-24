@Library('xmos_jenkins_shared_library@v0.14.1') _


pipeline {
    agent none

    parameters {
        string(
            name: 'TOOLS_VERSION',
            defaultValue: '15.0.1',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        booleanParam(
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
        )
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Rebuild and push a new docker image'
        )
    }

    options {
        //buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage ("Build and Push Image") {
            when {
                anyOf {
                    changeset 'Dockerfile'
                    expression { return params.PUSH_IMAGE }
                }
            }
            agent {
                label 'docker'
            }
            steps {
                script {
                    def image = docker.build('xmos/ai_tools')
                    docker.withRegistry('https://docker-repo.xmos.com', 'nexus') {
                        image.push(GIT_BRANCH)
                        // if on master/release/etc:
                        //   push version/hash
                    }
                }
            }
        }
        stage ("Pull and Use Image") {
            agent {
                docker {
                    image 'xmos/ai_tools:${GIT_BRANCH}'
                    registryUrl 'https://docker-repo.xmos.com'
                    alwaysPull true
                }
            }
            stages {
                stage("Setup") {
                    steps {
                        sh "rm -rf *"
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
                        sh "conda env create -q -p ai_tools_venv -f environment.yml"
                        sh "/XMOS/get_tools.py " + params.TOOLS_VERSION
                    }
                }
                stage("Update all packages") {
                    when { expression { return params.UPDATE_ALL } }
                    steps {
                        sh "conda update --all -y -q -n ai_tools_venv"
                    }
                }
                stage("Build") {
                    steps {
                        sh "cat Makefile"
                        sh """pushd /XMOS/tools/${params.TOOLS_VERSION}/XMOS/xTIMEcomposer/${params.TOOLS_VERSION} && . SetEnv && popd &&
                              conda run -p ai_tools_venv make ci"""
                        junit "**/*_junit.xml"
                    }
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }
    }
}
