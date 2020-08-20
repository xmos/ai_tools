@Library('xmos_jenkins_shared_library@v0.14.1') _


pipeline {
    agent none

    stages {
        stage ("Build and Push Image") {
            agent {
                label 'docker'
            }
            steps {
                script {
                    def image = docker.build('xmos/ai_tools')
                    docker.withRegistry('https://docker-repo.xmos.com', 'nexus') {
                        image.push('0.1 latest')
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
