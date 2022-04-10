# Yohan John Lee's MLOps Workflow
## 1. Introduction
### 1.1	Overview

This MLOps workflow covering Deep Learning model prediction written with Tensorflow is intended to be used as a reference for other developers. The workflow is based on the following:
- DL Notebooks: A set of notebooks for Deep Learning model prediction
- DL Models: A set of models for Deep Learning model prediction
- Data preprocessor: SKlearn pipeline for data preprocessing
- Azure Static Web Apps
- Azure Functions
- Github Actions (CICD pipeline)
- Github Codespace (devcontainer & Dockerfile)


## 2 Logical Architecture
### 2.1	Logical System Component Overview
![Figure 1: Logical Architecture Overview](./.images/workflow.png)
- Azure Static Web Apps gets deployed via IaC (Azure Bicep)
- Code gets pushed to Github via integrated into the pipeline
- [DL Notebooks](#./dl/dl-ann.ipynb) gets trained and deployed to Azure Functions

## 3 Prediction Link
Ref: [DL](https://yellow-dune-08d318d10.1.azurestaticapps.net/)

## 3 MSDocs Link
Ref: [MSDocs](https://docs.microsoft.com/en-us/azure/static-web-apps/functions-bring-your-own/)