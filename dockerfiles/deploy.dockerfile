#Docker file for deploying the model
FROM docker.io/albertgarde/base:latest
EXPOSE 8080 

COPY models/model_quick models/model
COPY frontend/ frontend/

ENTRYPOINT ["uvicorn", "--port", "8080", "--host", "0.0.0.0", "mlopstinystories.serve:app"]