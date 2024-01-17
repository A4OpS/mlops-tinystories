---
layout: default
nav_exclude: true
---

# A short guide to our docker container.
If you have questions to this guide contact Andreas Raaskov.

## Automatical build
We have tried to automate the workflow of our docker containers but at the times of this writing the task is not done. 

Running the bash script:
BuildPush.bash should automatically build containers and push them to your docker repository. In order to push too your docker repository make sure this folder has a `.env` file with the following format:

```
DOCKER_USERNAME=<your docker username> 
DOCKER_TOKEN=<your docker token>
```

In the github action folder there is also a yaml setup for a github action that requres you docker credentials is a github secret.

## Manual build
In order to make a container you first need to create the base container by running:
```
docker build --file dockerfiles/base.dockerfile --tag base . \
```
For a CPU container. For a GPU container run:
```
docker build --file dockerfiles/base-cuda.dockerfile --tag base . \
```

Sometimes this command fails if you don't have the NVIDA images to get it try running

```
Docker pull nvcr.io/nvidia/pytorch:23.12-py3 
```

Then you can build the train image and deploy image useing:

```
docker build --file dockerfiles/trainer.dockerfile --build-arg BASE_IMAGE=base\
    --tag docker.io/<DOCKER_USERNAME>/<IMAGE_NAME>:<Train_TAG> .
```

## Getting containers
Prebuild images for this project can be found on:

https://hub.docker.com/repository/docker/andreasraaskovdtu/mlops-tinystories/general

for CPU images and:

https://hub.docker.com/repository/docker/andreasraaskovdtu/mlops-tinystories-cuda/general

for GPU images.

## Running tain container.

A fully functional train container pulled from the repositor should run with the command.  

```
docker run --rm --gpus all -e WANDB_API_KEY=<your WANDB API key>  andreasraaskovdtu/mlops-tinystories:trainer --config-name <name of a config file> <optinal hydra paremeters>
```

If running without a GPU delete 
```
--rm --gpus all
```

And if you made your own repository replace: 
```
andreasraaskovdtu/mlops-tinystories:trainer 
```

## Running interactively
For debugging make an interactive container by commenting out the last line in the docker build (remove the entry point).

You can now open the container whith the inteactive flag `-it`.

```
docker run --rm --gpus all -it  andreasraaskovdtu/mlops-tinystories:trainer
```

Now you can run the train script as if it was a computer. 