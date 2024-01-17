# This is a basic dockerfile used for debugging and manual testing/trainning

#How to run this dockerfile
# 1. Ensure that docker is running
# 2. Build the container: `docker build -f dockerfiles\base.dockerfile . -t base:latest`
# 3. Run the container in interactive mode: `docker run --gpus all -it --entrypoint sh base:latest`

# Get base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#Copy data/ make sure all files are on the device 
#ToDo: Integate with DVC 
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlopstinystories/ mlopstinystories/


# Install python dependencies
WORKDIR /

#Command that install dependencies in the requirements.txt file
#RUN pip install -r requirements.txt --no-cache-dir

#Command that install dependencies in the requirements.txt file and cache them
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

#Command that install the project in the current directory (pyproject.toml)
RUN pip install . --no-deps --no-cache-dir
