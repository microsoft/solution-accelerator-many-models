# 1 - Setup Pipeline

The setup pipeline will:

- Deploy Azure Machine Learning and the other necessary resources into the resource group you specified.

- Set up the Azure Machine Learning worskpace, creating a compute target and attaching the AKS cluster (if needed).

- Download as many files as you specified in the `DATASET_MAXFILES` variable in the [variable group](../README.md/#2-create-variable-group), and register them as a dataset in AML.

## Instructions

Create the pipeline as explained in [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline), selecting branch **``v2-preview``** and setting the path to [`/mlops-pipelines/1-setup/pipeline-setup.yml`](pipeline-setup.yml).

## Result

The pipeline run should look like this:

<img src="../../images/mlops_pipeline_1_setup.png"
     width="1000"
     title="Setup Pipeline"
     alt="Stages and jobs as described below" />

Containing the following stages and jobs:

- Deploy Infrastructure
  - IaC Build
  - IaC Deployment
- Environment Setup
  - Deploy AML Compute
  - Attach AKS cluster to AML
- Data Preparation
  - Sample Files Setup
  - Register Dataset

## Details

If you want to customize the Azure resources deployed, you can modify the ARM templates and parameters under [`arm-templates/`](arm-templates).

AKS cluster will be attached only if the variables related to deploying in AKS are set (see [the section about deployment](../2-modeling/README.md#deployment) in the modeling pipeline for details).
