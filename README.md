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

In this project, I build a [Wide and Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) neural network to predict whether the user will click on the app. I use the PyTorch framework to build and train the model.
Due to the high number of records in the dataset (40428967 in the training set alone), I used PySpark to extract features from the raw dataset and then fed them to the neural network model.

The spark job was wrapped up with [spark operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator) and deployed on GKE on the Google Cloud Platform.

- The Dockerfile for the image used for the spark job as well as the python file are located in `/docker/`.
- The base image can be found [here](https://github.com/yinanli617/pyspark-gcp) which is configured to use BigQuery and Cloud Storage from Google Cloud Platform.
- The spark job can be run by `kubectl create -f ctr-spark-job.yaml -n <namespace>`.

**Note: if you would like to try out this by yourself, make sure to change the settings in the yaml file to use your own `<project-name>`, `<service-account>`, `<namespace>`, and `<gs-bucket>` etc.**

## Feature engineering

1. For wide features:

   - All fields with `nunique < 100` are used;
   - The following cross-product features are constructed:
     - `hr` and `device_type`;
     - `device_type` and `app_category`;
     - `device_type` and `site_category`;
     - `banner_pos` and `device_type`;
   - The "single" features and the cross-product features are one-hot encoded to generate the wide features (dimension: 475).

2. For deep features:
   - Embeddings of the following features are generated and fed to the deep part of the model:
     - `device_model` (embedding_dim: 256);
     - `app_id` (embedding_dim: 256);
     - `site_id` (embedding_dim: 256);
     - `site_domain` (embedding_dim: 256);
     - `app_domain` (embedding_dim: 128);

## Model architecture

- The deep part of the model contains 3 hidden layers with `hidden_size = [512, 256, 128]`;
- Relu is used as activation function in the deep part;
- The original [paper](https://arxiv.org/abs/1606.07792) used L1 regularization for the wide part. Here, Adam optimizer is used for both wide and deep parts. However, a dropout layer with `dropout_p=0.7` is added to the wide part before merging with the deep part.

## Results

ROC_AUC was selected as the evaluation metric (see the notebook `compute_roc_auc.ipynb` for the reasons). After 15 epochs of training, the wide and deep model results in ROC_AUC of 0.7497 - a significant improvement from gradient boosting tree model trained in Spark which has an ROC_AUC score of 0.7112.
