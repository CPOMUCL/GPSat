pipeline {

    //example pipeline build with 
    
    agent any  // This tells Jenkins to use any available agent to run the pipeline

    environment {
        // Define environment variables if needed, for example:
        VENV = 'venv'
    }

    stages {

        stage('Setup Environment') {
            steps {
                echo 'Setting up virtual environment...'
                script {
                    // Check if the virtual environment exists, create it if it doesn't
                    if (!fileExists("${VENV}")) {
                        sh 'python3 -m venv ${VENV}'
                    }
                    // Install or update required packages
                    sh '${VENV}/bin/pip install -r requirements.txt'
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                // run tests form venv using pytest
                sh '${VENV}/bin/pytest tests'
            }
        }

        stage('Run Modules in Bash Script') {
            steps {
                echo 'Activating virtual environment'
                sh '. venv/bin/activate'
                echo 'Running Shell Script'
                // run shell script containing default examples
                sh './run_examples.sh'
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            // Add any post-build cleanup steps here
            cleanWs()  // This cleans up the workspace after the build is done
        }

        success {
            echo 'Build was successful!'
            // Any post-build actions for successful builds can be added here
        }

        failure {
            echo 'Build failed.'
            // Any post-build actions for failed builds can be added here
        }
    }
}

