# 1 - Setup Pipeline - Details

## Deploy Infrastructure

The first stage of the pipeline will deploy into Azure all the necessary infrastructure to run the Many Models Solution Accelerator.

If you want to customize the Azure resources deployed, you can modify the ARM templates and parameters under [`arm-templates/`](arm-templates).

## Environment Setup

During the environment setup:

- A compute target will be created to run training and forecasting.
- An existing AKS cluster will be attached for model deployment, if defined. This will only happen if the variables related to deploying in AKS are set (see [the section about deployment](../2-modeling/Details.md#deployment-optional) in the modeling pipeline for details).

## Data Preparation

The pipeline will download as many files as you specified in the `DATASET_MAXFILES` variable in the [variable group](../#2-create-variable-group), split them, and register them as train and inference datasets in AML.
