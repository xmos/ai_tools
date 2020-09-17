@Library('xmos_jenkins_shared_library@v0.14.1') _


pipeline {
    agent none

    parameters {
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
                                          parentCredentials: true,
                                          recursiveSubmodules: true],
                                         [$class: 'CleanCheckout']],
                            userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                                 url: 'git@github.com:xmos/ai_tools']]
                        ])
                        sh "conda env create -n .venv -f environment.yml"
                    }
                }
                stage("Update all packages") {
                    when { expression { return params.UPDATE_ALL } }
                    steps {
                        sh "conda update --all -n .venv"
                    }
                }
                stage("Check") {
                    steps {
                        sh "conda run -n .venv python -c 'import tensorflow'"
                        sh """#!/bin/bash -l
                              xcc --version"""
                    }
                }
                stage("Build") {
                    steps {
                        sh """#!/bin/bash -l
                              conda run -n .venv make ci > make_output.txt"""
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
