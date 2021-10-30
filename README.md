# Click Through Rate Prediction with Wide and Deep Neural Network

## Introduction
In this project, I show how to combine PySpark (and running spark jobs on the powerful [Kubernetes](https://kubernetes.io) system) and PyTorch to do feature engineering over a huge dataset and train a wide and deep model to predict whether a user will click on an app or not.

Click through rate (CTR) prediction is crucial for online advertising company to provide better services. Some real-world applications of CTR prediction includes:
- Recommending/showing movies that the user will most likely be interested in clicking on (and probably watching) on their homepage for online streaming platforms;
- Recommending/showing applications that the user will most likely click on (and install) in mobile phone application stores;
- Showing the advertisements that the user will most likely be interested in click on when they are browsing the webpages;

## Dataset
The dataset I used in this project is the famous [Avazu CTR dataset](https://www.kaggle.com/c/avazu-ctr-prediction) that was from a Kaggle competition.

From the overview on the Kaggle page, the data fields in this dataset are:
- id: ad identifier
- click: 0/1 for non-click/click
- hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
- C1 -- anonymized categorical variable
- banner_pos
- site_id
- site_domain
- site_category
- app_id
- app_domain
- app_category
- device_id
- device_ip
- device_model
- device_type
- device_conn_type
- C14-C21 -- anonymized categorical variables

From this, we can see that all features are categorical, and the target value we want to predict is `click` which is a binary classification problem.

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

## Model structure
- The deep part of the model contains 3 hidden layers with `hidden_size = [512, 256, 128]`;
- Relu is used as activation function in the deep part;
- The original [paper](https://arxiv.org/abs/1606.07792) used L1 regularization for the wide part. Here, Adam optimizer is used for both wide and deep parts. However, a dropout layer with `dropout_p=0.7` is added to the wide part before merging with the deep part.

## Results
