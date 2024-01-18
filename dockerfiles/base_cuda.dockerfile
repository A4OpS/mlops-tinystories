# This is a basic dockerfile used for debugging and manual testing/trainning

#How to run this dockerfile
# 1. ensure that docker is running
# 2. Build the container: docker build -f dockerfiles\base_cuda.dockerfile . -t cuda:latest
#If step 2 fails to autehnticate run the following command: docker run -it --rm nvcr.io/nvidia/pytorch:23.12-py3 bash
# 3. Run the container in interactive mode: docker run -it --entrypoint sh cuda:latest

# Get base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

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
#WORKDIR /

#Command that install dependencies in the requirements.txt file
#RUN pip install -r requirements.txt --no-cache-dir -v

#Command that install dependencies in the requirements.txt file and cache them
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir

#Command that install the project in the current directory (pyproject.toml)
RUN pip install . --no-deps --no-cache-dir -v

