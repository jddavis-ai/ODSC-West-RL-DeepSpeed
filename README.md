### ODSC-West 2022 Tutorial on MPI and DeepSpeed

This tutorial covers absolute beginner (MPI), intermediate (summarization and translation tasks) and advanced (a computer vision example).  We also provide supplementary materials.  We do not recommend working on any supplementary materials during the workshop as you will not have access to appropriate GPU clusters.  However you can take this code and run it on any MPI + DeepSpeed Cluster of your choice. Of course we hope you choose Domino!  Below are the dockerfile instructions for running the MPI+DeepSpeed code on your own.

### Abstract
As Deep Learning and AI use grows, complexity and size of the models grows. Training large models such as GPT-2 and Megatron, among others, has been a daunting task. Several distributed computing frameworks are available to address these tasks, and the oldest and most resilient is the OpenMPI (open message passing interface) library.

OpenMPI is used for high-performance computing at supercomputing centers as part of distributed computing systems. In the first half of this workshop, we will get to know more about OpenMPI and work with it hands-on using a python interface. The second half of the workshop will focus on using DeepSpeed on OpenMPI and work through a few examples. Examples for MPI basics will include inferring pi using distributed computing and running python scripts using OpenMPI.

In 2020 the library, DeepSpeed, was developed to train huge models using OpenMPI. DeepSpeed is a flexible library that can aid in hyperparameter tuning and transfer learning and substantially integrates with the transformers library. For training large sequence-to-sequence models, DeepSpeed is at the top of the heap. Examples we will go over using DeepSpeed include translation from English to Romanian, a transfer learning example for the summarization of the CNN/Daily Mail dataset, and a proteomics example.

We will discuss the benefits of using OpenMPI and DeepSpeed and when not to use them. We present other examples of distributed computing and compare them to MPI with DeepSpeed.

Background Knowledge Required for Participants:
Little to no knowledge of cpp, intermediate knowledge of python, No familiarity with DeepSpeed is required, Some exposure to neural network


### Biosketch

Jennifer Davis, Ph.D. is a Staff Field Data Scientist at Domino Data Labs, where she empowers clients on complex data science projects. She has completed two postdocs in computational and systems biology, trained at a supercomputing center at the University of Texas, Austin, and worked on hundreds of consulting projects with companies ranging from start-ups to the Fortune 100. Jennifer has previously presented topics at conferences for Association for Computing Machinery on LSTMs and Natural Language Generation and at conferences across the US and in Italy. Jennifer was part of a panel discussion for an IEEE conference on artificial intelligence in biology and medicine. She has practical experience teaching both corporate classes and at the college level. Jennifer enjoys working with clients and helping them achieve their goals.

#### Dockerfile Instructions for the Workspace

```
# System-level dependency injection runs as root
USER root:root

# Validate base image pre-requisites
# Complete requirements can be found at
# https://docs.dominodatalab.com/en/latest/reference/environments/Automatic_Custom_Image_Compatibility.html#custom-image-requirements-for-domino-compatibility
RUN /opt/domino/bin/pre-check.sh

# Configure /opt/domino to prepare for Domino executions
RUN /opt/domino/bin/init.sh

# Validate the environment
RUN /opt/domino/bin/validate.sh

RUN apt-get update
RUN apt-get -y install pdsh

RUN pip install deepspeed mpi4py pytorch-lightning rouge_score sentencepiece sacrebleu datasets pypdsh ipywidgets IProgress jupyter evaluate torchvision pillow>=7.1.0 matplotlib
RUN pip install git+https://github.com/huggingface/transformers accelerate

```

#### Dockerfile Instructions for the Cluster

```
USER root

RUN apt-get update
RUN apt-get -y install pdsh
RUN pip install deepspeed pypdsh mpi4py pytorch-lightning rouge_score sentencepiece sacrebleu datasets evaluate torchvision pillow>=7.1.0 matplotlib
RUN pip install git+https://github.com/huggingface/transformers accelerate
```

Please feel free to reach out to the Field Data Science Team with any questions.  We appreciate your attendance at our workshop. 
