#This file requires the base image to be built first
ARG BASE_IMAGE
FROM $BASE_IMAGE

#Make and output dir: remember to mount this dir to the host using docker run -v /outputs:/outputs mlops-tinystories:
RUN mkdir outputs/

COPY conf/ conf/

COPY dockerfiles/train_entry.sh . 
RUN chmod +x train_entry.sh

# dvc
RUN dvc init --no-scm
COPY .dvc/config .dvc
RUN dvc config core.no_scm true
COPY .dvcignore .
COPY data.dvc .


# ENTRYPOINT ["python", "-u", "mlopstinystories/train_model.py"]
ENTRYPOINT [ "/train_entry.sh" ]
# CMD ["python", "-u", "mlopstinystories/train_model.py"]