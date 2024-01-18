---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 42

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s194323, s194368, s183901, s183969, s194248

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the transformers package from the third-party framework huggingface. From this framework we used the pre-defined architectures for the language models. We can easily initialize a model with the same architecture as from the tiny stories paper and try to train it from scratch. This framework also allowed us to change some parameters in the achitecture like the number of layers, the vocabulary size and more. We specifically used the GPTNeoConfig to define the parameters of the model and passed this into the GPTNeoForCasualLM to initialize the model. These functionalities from the transformers package saved us a lot of time on getting a first model set up and working properly, giving us more time to focus on the Operations and training parts.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

In our GitHub repository we worked on different branches when implementing new features to the project. Each member would add dependies to requirements.txt and requirements_dev.txt. These two files are then used to keep track of dependencies needed for a production enviroment and dependencies for a development enviroment respectively. A new member to the team could then access the needed branch and install all dependencies in the two files. So after having created and activated a new enviroment to work in, a new team member could run the two following commands to get all needed dependencies:
pip install -r requirements.txt
pip install -r requirements_dev.txt

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

For this project we have used the suggested cookiecutter template. We have structured all code in the project under the assumption that the working directory should always be the root of the template, namely mlops-tinystories. In the mlopstinystories subfolder, we have put all python code to download and process data, define and initialize our model, train a model, make predictions for the model. The scripts that download and process the data save it in the mlops-tinystories/data folder in raw and processed subfolders.
We have added the conf folder we save configurations uses via the Hydra package and in the tests folder we have saved the python scripts to run unit-tests. We have not used the notebooks folder as we have not worked with jupyter notebooks.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

As part of continuous integration, we added a GitHub action that checks all commits to main via the ruff package. So if a team member tried to commit some python files that did not follow ruff rules, then their commits/pull requests to main would be failed and they would need to commit again once their scripts follow ruff rules.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have over 30 unittests implemented in three files. We test the data, the model and if the model can be trained. We test when getting the raw data and if the data is processed correctly. For getting the raw data, we test if all relevant directories exist and have been created properly. We also test if all raw files actually contain some data. For processing the data we test if a dataloader is created and also test that it is not empty. The model is also tested, to ensure the input and output is the correct format, and if the model can calculate gradients. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The measured code coverage we got is 90%. This includes all of the files that are needed to run the tests, which means there are some python files in our repository that is not included in the coverage percentage. We are a little bit away from 100%. We found that when writing unit tests, then we kept on thinking of new ways to test our code. We ended up at some point not writing anymore unit tests. So even though we have a high code coverage, then the fact that we did not have enough time to write all the unit tests we wanted shows that there is still a chance that our code could contains errors.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used both brances and pull requests in our project. At the beginning of the project, an issue was made for all items seen on the project checklist. When a team member would start working on an issue they would create a new branch to work on. The issue would also be assigned to that team member to help us keep track of who works on what. When the issue/checklist item was done and working, then the team member would commit and create a pull request in order to merge that branch with the main branch. We set up a rule that one other team member has to review and accept that pull request before the merge is done. When and if the request was accepted and merged into main then the issue could also be closed.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did implement dvc but we did not use it for managing our data. The data for this project was downloaded using an API and when training a model we could alter the hydra configurations to control how much data was used in training and validation. We only used dvc to share model files needed for deployment.
Through the hydra configurations we were able to have a lot of control of what data we would train on. If this was not the case, then dvc could have helped us keep track of different versions of data we trained on.
Our entire dataset takes up 2 GB's of space, which meant where ever we ran our training all the data would be able to be kept in memory. If this was not the case, then more use of dvc to manage the data would probably have been usefull.


### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have 3 seperate files for our CI. The first one we included was ruff for every merge to main branch. To avoid introducing bugs into the main branch we also included the test should be passed before a merge to main. We also run pytest when trying to merge into main where different parts of the data is tested. Here again all tests should pass before a merge into main is possible.
An example of a triggered workflow can be seen here: <https://github.com/A4OpS/mlops-tinystories/actions/runs/7568697468>*
We tried to do continues integration of docker containers using github actions and mangaed to write a DockerBuild.yaml file that integrates with github action to automatically build a docker image and push it to a dockerhub that can be accessed through: 
`docker pull andreasraaskovdtu/mlops-tinystories:6a9208c034d2188ff4d0a6157efef563e7805a73`  

Unfortunately the free compute at Github could not handle the CUDA container due to lack of space. We tried instead to set up the automatic building of containers in GCP.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used config files for our experiments, where we choose to handle it with hydra. The config file name was then included was as an argparser. It worked the following way:
python train_model --config-name quick
where the different configs had a folder making it easier to log what experiments that had been performed. These config files gave us control over how much data to train and validate on, the hyperparameters in the model architecture and different parameters in our training loop.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

For this a combination of weights and biases and hydra was used. Weight and biases was used to keep track of how the learning progressed, namely how the training and validation loss progressed during training. Hydra would save the configuration in a generated folder based on date and timestamp of training. In W&B one could then compare the train and validation losses from the different experiments to find the better performing experiments.
To reproduce an experiment, a team member could compare the stimestamps from W&B and hydra, find the relevant hydra folder and get the configuration from that yaml file.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We have included the log from W&B from one of our experiments in [this figure](figures/W&B.png). This is the experiment for the deployed model. In the figure four graphs are shown. The two most important graphs are two most left ones in the top. In the graph seen in the top row all to the left, we logged the validation loss for the experiment. In this experiment it was calculated and logged every 200 steps. In the graph seen in the top row second most to the left, we have logged the training loss every 5 steps. With the training loss it is important to check if it is decreasing and converging. We see that the training loss converges quite quickly and afterwards the optimizer is stuck in some minimum.
Comparing the training loss and the validation loss is also important. If the training loss kept decreasing while there was no improvement in the validation loss, then this would suggest that the model is overfitting.
The last two graphs show how long we are in the experiment, tracking which epoch we are executing in each step and how many times we have logged data compared to the number of steps in the training.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We relatively quickly got docker containers running on CPU, however, we spend a lot of time on the CUDA docker container since each container was around 26 GB which made them hard to work with. In order to save download speed we made a bash script that first created a base container with all the projects dependencies, then we used the base container to create a train container that had access to the data set and the config files in order to rapidly train new models. To run the docker container for training one had to include the config file as an argparse:
docker run path:trainer --config-name <name>
We also created a deploy container that can run the model, which was also used to deploy the model on GCP.
One of the docker files can be found here: <https://hub.docker.com/layers/andreasraaskovdtu/mlops-tinystories/da094e9ae607afc3d5217438d4f93ae3c5f32ab9/images/sha256-749a7a1e2c3883d01e4234fee92db3ff663182eb4430ec56a9adc57ef61feb15?context=explore>*

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Due to using lightning and transformers framework, very little debugging was needed when running the code locally. In that case simple print statements was used to fix simple bugs.
A PyTorchProfiler was wrapped around our training loop to analyze if any optimization was needed. After inspecting the time spent on certain steps in the training, it was found that there was immediate need for any optimizations. Much of our code is setup by using different frameworks like lightning and transformers package from huggingface. Since most of our code relies on these well established frameworks, it does not come as a suprise that our was already somewhat optimized.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

In GCP we used the compute engine to run experiments on a GPU. By cloning the repo into the VM instance we were able to save the different configurations and track the progress of the experiment on W&B.
We also used the Bucket on GCP to keep the data and the models on there. The idea is then to keep the best model that has been trained in the bucket.
We also used the Cloud Functions services to set up deployment of the best model.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to interactively run experiments on the VM instances. We used VM instances with the Tesla v100 GPU. We tried to set up so we could run different experiments using the compute enginge, however our training script would often stop working/running without any error or warnings messages. This made debugging extremely hard. Especially considering that we could run our training script on a local laptop GPU. This also resulted in us not having enough time to find a configuration that produced a working model.
Due to problems with getting our containers functioning properly, we also did not have the time to test running containers in the compute engine.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Bucket figure](figures/bucket.png) 

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We did manage to deploy our model locally serving the model locally and also testing it with a few prompts. Afterwards we started on trying to deploy the model on the cloud...

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

s194323 used 9 credits in trying to get the compute engine to run experiments.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Overall, the biggest challenge with this project for our group was everything regarding deployment and the cloud. From trying to build containers to having our containers and our code and dvc collaberating with GCP.
We had a lot of problems building the containers and also making sure they worked afterwards by trying to run the images on other computers. The reason for this was that building the containers took very long to build, which meant that debugging this process took a very long time.
For running experiments on the compute engine, we also ran into trouble with our experiments dying or running into a wall without any proper warnings/errors. This made finding the reason for why the VM's could not run our code very hard. The troubles of using this compute engine also meant we were not able to run large experiments quickly.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s194323 (Aleksander) was in charge of setting up the profiler and analyzing the output to see if any optimizations were needed. This student also tried running experiments on the compute engine in GCP.
Student s194368 (Alexandra) was in charge of developing unit tests for the data, for the model and for the training.
Student s183969 (Albert) was in charge of setting up the cookiecutter template, setting up the data, model and training script. The student also set up the CI in our GitHub repository, integrated hydra configurations into our training code.
Student s183901 (Andreas) was in charge of setting up our docker containers and figuring out how to implement it as a part of CI in GCP.
Student s194248 (Simon) was in charge of settting up the dvc and afterwards setting up dvc in GCP.
Everybody contributed to: reviewing pull requests on github, adhering to coding practices, different parts of deploying on the cloud.