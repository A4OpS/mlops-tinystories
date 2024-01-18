#Docker file for deploying the model
ARG BASE_IMAGE
FROM $BASE_IMAGE

EXPOSE 8080 
COPY frontend/ frontend/

COPY dockerfiles/deploy_entry.sh . 
RUN chmod +x deploy_entry.sh

# get dvc
RUN dvc init --no-scm
COPY .dvc/config .dvc
RUN dvc config core.no_scm true
COPY .dvcignore .
COPY models.dvc .


ENTRYPOINT [ "/deploy_entry.sh" ]
CMD ["uvicorn", "--port", "8080", "--host", "0.0.0.0", "mlopstinystories.serve:app"]
# ENTRYPOINT ["uvicorn", "--port", "8080", "--host", "0.0.0.0", "mlopstinystories.serve:app"]