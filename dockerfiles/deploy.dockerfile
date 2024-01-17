#Docker file for deploying the model
ARG BASE_IMAGE
FROM $BASE_IMAGE

# ToDO : add data integartion with DVC
COPY data/ data/

ENTRYPOINT ["python", "-u", "mlopstinystories/predict_model.py"]