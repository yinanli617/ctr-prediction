# Click Through Rate Prediction with Wide and Deep Neural Network

## Introduction

In this project, I show how to build a machine learning workflow and deploy it on [Kubernetes](https://kubernetes.io) to load a large dataset (40 million+ records), preprocess it and perform feature engineering, and finally train a wide and deep model and evaluate model performance using a separate test dataset.

There are three main objectives in this project:

1. Train a SOTA [Wide and Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) that combines the strength of _memorization_ and _generalization_ to make click-through rate (CTR) prediction (for an introduction of CTR prediction, check out [this article](https://content-garden.com/click-through-rate-prediction)).

2. Demo a) how to leverage distributed computing frameworks like Apache Spark to handle big datasets, perform ETL efficiently, and generate features that can be used for downstream ML tasks; b) how to use deep learning frameworks like PyTorch to build a neural network model architecture from scratch, train it with a GPU accelerator, and then evaluate the model performance using a separate dataset.

3. The advent of cloud computing greatly facilitates ML workflows. Both the spark job and the pytorch job are encapsulated in Docker containers and deployed on a multi-nodal Kubernetes cluster, using the [spark-on-k8s-operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator) and the [pytorch operator](https://github.com/kubeflow/pytorch-operator) respectively. The deployment of a complete ML workflow on Kubernetes clusters substantially improves cost efficiency and time efficiency, proves scalable when handling large workloads, and makes the containerized models easy to deliver.

## Dataset

The dataset I used in this project is the [Avazu CTR dataset](https://www.kaggle.com/c/avazu-ctr-prediction) from Kaggle. The dataset contains categorical raw features and a binary label column. For more details about the dataset and the rationale behind feature engineering in this project, check out this [notebook](https://github.com/yinanli617/ctr-prediction/blob/master/pytorch-wide-and-deep.ipynb).

## Approach

### Step 1 - Feature engineering with Apache Spark

The Spark job [python script](https://github.com/yinanli617/ctr-prediction/blob/master/pyspark_docker/pyspark-ctr.py) essentially handles all the ETL work and feature engineering. The raw data is stored in a distributed file system (GCS in this project) which is loaded in the spark job. The single wide features and cross-product are one-hot encoded and the wide features are label encoded (to get embeddings later during training). The generated features are saved on GCS.

The python script is stored in a Docker container and will be executed when the K8s pods run the container. The Dockerfile of the spark job image can be found [here](https://github.com/yinanli617/ctr-prediction/tree/master/pyspark_docker). The Dockerfile of the base image can be found in [this repository](https://github.com/yinanli617/pyspark-gcp) and is an adaptation from the [official spark operator image](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/master/docs/gcp.md), which is configured to run on Google Cloud Platform.

The spark job configuration is [here](https://github.com/yinanli617/ctr-prediction/blob/master/k8s_jobs/ctr-spark-job.yaml) and the specs can be modified easily to scale up the requested resources. Due to limited quota, I used 2 executors each with 3 cores.

To start the Spark job, run `kubectl create -f ./k8s_jobs/ctr-spark-job.yaml`

### Step 2 - Training wide and deep model with PyTorch

Just like the Spark job, the [python script](https://github.com/yinanli617/ctr-prediction/blob/master/pytorch_docker/wide_deep_k8s.py) of the PyTorch job is stored in a container built with the simple [Dockerfile](https://github.com/yinanli617/ctr-prediction/blob/master/pytorch_docker/Dockerfile).

Due to limited quota, I used a single GPU node as the master node and `num_workers=4`. In production, this can be easily scaled up by modifying the [PyTorch job configuration](https://github.com/yinanli617/ctr-prediction/blob/master/k8s_jobs/ctr-pytorch-job.yaml).

To start the PyTorch job, run `kubectl create -f ./k8s_jobs/ctr-pytorch-job.yaml`

Checking the logs of the running pods should show you something like below:
<br/><br/>
![pytorch-job-log](https://github.com/yinanli617/ctr-prediction/blob/master/gif/pytorch-job-ctr.gif)

## Results

A step-by-step notebook that runs the training on a local GPU is also provided in this repository. When training locally, the best logloss from the test dataset is 0.3958. The deployed PyTorch job on Kubernetes used different batch size which results in a better logloss of 0.3946.

ROC_AUC was selected as the evaluation metric (see the notebook `compute_roc_auc.ipynb` for the reasons). After 15 epochs of training on a local GPU, the wide and deep model results in ROC_AUC of 0.7497 - a significant improvement from gradient boosting tree model trained in Spark which has an ROC_AUC score of 0.7112.
