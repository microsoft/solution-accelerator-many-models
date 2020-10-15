# 1 - Setup Pipeline

The setup pipeline will:

- Deploy Azure Machine Learning and the other necessary resources into the resource group you specified.
- Set up the Azure Machine Learning worskpace.
- Download as many files as you specified in the `DATASET_MAXFILES` variable in the [variable group](../#2-create-variable-group), and register them as a dataset in AML.

## Instructions

Create the pipeline as explained in [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline), selecting branch **``v2-preview``** and setting the path to [`/mlops-pipelines/1-setup/pipeline-setup.yml`](pipeline-setup.yml).

Then, run the pipeline and wait for it to finish.

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

If you want to learn about the details and ways to customize the setup pipeline please read the [details page](Details.md).
