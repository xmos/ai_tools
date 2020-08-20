@Library('xmos_jenkins_shared_library@v0.14.1') _


pipeline {
    agent none

    parameters {
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Whether to rebuild and push a new docker image'
        )
    }

    stages {
        stage ("Build and Push Image") {
            when { expression { return params.BUILD_IMAGE }}
            agent {
                label 'docker'
            }
            steps {
                script {
                    def image = docker.build('xmos/ai_tools')
                    docker.withRegistry('https://docker-repo.xmos.com', 'nexus') {
                        image.push('latest')
                    }
                }
            }
        }
        stage ("Pull and Use Image") {
            agent {
                docker {
                    image 'xmos/ai_tools:latest'
                    registryUrl 'https://docker-repo.xmos.com'
                    alwaysPull true
                }
            }
            steps {
                sh "conda activate xmos && python -c 'import tensorflow; print(dir(tensorflow))'"
                sh "make all"
            }
        }
    }
}
