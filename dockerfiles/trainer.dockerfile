# Docker file for training the model

#This file requires the base image to be built first
ARG BASE_IMAGE
FROM $BASE_IMAGE

#Make and output dir: remember to mount this dir to the host using docker run -v /outputs:/outputs mlops-tinystories:
RUN mkdir outputs/

#Copy data/ make sure all files are on the device
COPY data/ data/
COPY conf/ conf/

# ToDO : add data integartion with DVC

ENTRYPOINT ["python", "-u", "mlopstinystories/train_model.py"]