#Docker file for deploying the model
ARG BASE_IMAGE
FROM $BASE_IMAGE

# ToDO : add data integartion with DVC
COPY data/ data/
EXPOSE 8080 

ENTRYPOINT ["uvicorn", "--port", "8080", "--host", "0.0.0.0", "mlopstinystories.serve:app"]