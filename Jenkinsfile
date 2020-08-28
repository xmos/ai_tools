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

    stages {
        stage ("Build and Push Image") {
            when {
                anyOf {
                    changeset pattern: "(environment.*\\.yml)|(Dockerfile)", comparator: "REGEXP"
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
                stage("Create env") {
                    steps {
                        sh "conda env create -p .venv -f environment.yml"
                        sh "conda list"
                        sh "conda list -p .venv"
                        withEnv(["CONDA_PREFIX=.venv"]) {
                            sh "conda list"
                        }
                    }
                }
                stage("Update all packages") {
                    when { expression { return params.UPDATE_ALL } }
                    steps {
                        sh """conda update --all -p .venv"""
                    }
                }
                stage("Install local package") {
                    steps {
                        sh """conda run -p .venv pip install -e ./tflite2xcore"""
                        sh """bash -lc 'xcc --version'"""
                        sh """xcc --version""" // check tools work
                    }
                }s
                stage("Check") {
                    steps {
                        sh """#!/bin/bash -l
                              conda list --export
                              python -c 'import tensorflow'"""
                    }
                }
                stage("Build") {
                    steps {
                        sh """#!/bin/bash -l
                              make all"""
                    }
                }
            }
        }
    }
}
