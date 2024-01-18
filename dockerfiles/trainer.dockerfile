# Docker file for training the model

#This file requires the base image to be built first
ARG BASE_IMAGE
FROM $BASE_IMAGE

#Make and output dir: remember to mount this dir to the host using docker run -v /outputs:/outputs mlops-tinystories:
RUN mkdir outputs/

#Copy data/ make sure all files are on the device
COPY conf/ conf/

# copy in the "control" file
COPY dockerfiles/entry.sh . 

# dvc
RUN dvc init --no-scm
COPY .dvc/config .dvc
RUN dvc config core.no_scm true
COPY .dvcignore .
COPY data.dvc .

RUN chmod +x entry.sh


# ENTRYPOINT ["python", "-u", "mlopstinystories/train_model.py"]
ENTRYPOINT [ "/entry.sh" ]
# CMD ["python", "-u", "mlopstinystories/train_model.py"]