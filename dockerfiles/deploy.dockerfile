#Docker file for deploying the model
ARG BASE_IMAGE
FROM $BASE_IMAGE

EXPOSE 8080 

COPY models/model_quick models/model
COPY assets/favicon.ico favicon.ico

ENTRYPOINT ["uvicorn", "--port", "8080", "--host", "0.0.0.0", "mlopstinystories.serve:app"]