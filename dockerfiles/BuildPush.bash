
#!/bin/bash

#Warning running this scripit with the cuda container requires 45 GB of free space on your machine

# Stop the script if any command fails
set -e


# Define variables

#The name of the image that will be built / the name of the docker repository
IMAGE_NAME="mlops-tinystories"

#The base container holde all dependencies for the and is used to build specific containers 
#base.dockerfile is the dockerfile for the base container and base_cuda.dockerfile is the dockerfile for the cuda base container
base_container="base_cuda.dockerfile"

#The final name of the container that will be pushed to the docker hub
Train_TAG="trainer"
deploy_TAG="deploy"

# If your variables are set correctly, you should not need to change anything below this line

echo "Should build images be pushed to Docker Hub? (y/n) (remember to set DOCKER_USERNAME and DOCKER_TOKEN in .env file)"
read push
push=$(echo "$push" | tr '[:upper:]' '[:lower:]')


if [ "$push" == "y" ]; then
    # Load  Docker Hub credentials from .env file .env file must contain DOCKER_USERNAME = <your username> DOCKER_TOKEN = <your-token>
    if [ -f dockerfiles/.env ]; then
        export $(cat dockerfiles/.env | xargs)
    else
        echo ".env file not found"
        exit 1
    fi
elif [ "$push" == "n" ]; then
    echo "Not pushing to Docker Hub"
    DOCKER_USERNAME="test"
else
    echo "Invalid input"
    exit 1
fi

echo "should containers be removed after pushing to docker hub? (y/n) (do this to save space)"
read remove
remove=$(echo "$remove" | tr '[:upper:]' '[:lower:]')


# Remove carriage return from variables (makes it work on Windows)
DOCKER_USERNAME=$(echo "$DOCKER_USERNAME" | tr -d '\r')
IMAGE_NAME=$(echo "$IMAGE_NAME" | tr -d '\r')
Train_TAG=$(echo "$Train_TAG" | tr -d '\r')

if [ $push == "y" ]; then
    # Login to Docker
    echo "$DOCKER_TOKEN" | docker login \
        -u "$DOCKER_USERNAME" --password-stdin docker.io
fi

#Build base container it saves time to build the base container first and then build the other containers from it
docker build --file dockerfiles/$base_container --tag base . \

echo "Base container built"

#Build the deploy container
docker build --file dockerfiles/deploy.dockerfile --build-arg BASE_IMAGE=base  \
    --tag docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$deploy_TAG .

echo "Deploy container built"

if [ $push == "y" ]; then
    #push the deploy container
    docker push docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$deploy_TAG
fi

if [ $remove == "y" ]; then
    #remove the deploy container
    docker stop docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$deploy_TAG
    docker rmi docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$deploy_TAG
fi

#Build the train container
docker build --file dockerfiles/trainer.dockerfile --build-arg BASE_IMAGE=base\
    --tag docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$Train_TAG .

if [ $push == "y" ]; then
    #push the train container
    docker push docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$Train_TAG
fi

if [ $remove == "y" ]; then
    #remove the train container
    docker stop docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$Train_TAG
    docker rmi docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$Train_TAG
fi

if [ $remove == "y" ]; then
    #remove the base container
    docker rmi base
fi