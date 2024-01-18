#!/bin/bash
set -e # Stop the script if any command fails


# The name of the image that will be built / the name of the docker repository
# IMAGE_NAME="mlops-tinystories-cuda"
IMAGE_NAME="mlops-tinystories"
DOCKER_USERNAME="test"

# The base container holde all dependencies for the and is used to build specific containers 
# base.dockerfile is the dockerfile for the base container and base_cuda.dockerfile is the dockerfile for the cuda base container
# base_container="base_cuda.dockerfile"
base_container="base.dockerfile"

# The final name of the container that will be pushed to the docker hub
train_TAG="train1.0"

docker build --file dockerfiles/$base_container --tag base .
echo "Base container built"


# # Remove carriage return from variables (makes it work on Windows)
# DOCKER_USERNAME=$(echo "$DOCKER_USERNAME" | tr -d '\r')
# IMAGE_NAME=$(echo "$IMAGE_NAME" | tr -d '\r')
# train_TAG=$(echo "$train_TAG" | tr -d '\r')

docker build --file dockerfiles/trainer.dockerfile --build-arg BASE_IMAGE=base\
    --tag docker.io/$DOCKER_USERNAME/$IMAGE_NAME:$train_TAG .
echo "Train container built"