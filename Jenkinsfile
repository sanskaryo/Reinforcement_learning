pipeline {
    agent { label 'Binod' }

    stages {
        stage("Code") {
            steps {
                echo "Cloning the repository"
                git url: 'https://github.com/KrishnaMittal-az/Reinforcement_learning.git', branch: 'main'
                echo "Repo cloned successfully"
            }
        }

        stage("Build") {
            steps {
                echo "Now building the Docker image"
                sh "docker build -t reinforcement_learning:v1 ."
                echo "Image build successful"
            }
        }

        stage("Image") {
            steps {
                echo "Pushing the image to DockerHub"
                withCredentials([usernamePassword(
                    credentialsId: "DockerHubCred",
                    usernameVariable: "dockerHubUser",
                    passwordVariable: "dockerHubPass"
                )]) {
                    sh "echo $dockerHubPass | docker login -u $dockerHubUser --password-stdin"
                    sh "docker image tag reinforcement_learning:v1 $dockerHubUser/reinforcement_learning:v1"
                    sh "docker push $dockerHubUser/reinforcement_learning:v1"
                }
            }
        }

        stage("Deploy") {
            steps {
                echo "Stopping any container using port 3000"
                sh "docker ps -q --filter 'publish=3000' | xargs -r docker stop"
                echo "Deploying the container"
                sh "docker run -d -p 3000:8501 $dockerHubUser/reinforcement_learning:v1"
                echo "Testing the CI/CD pipeline "
            }
        }
    }
}
