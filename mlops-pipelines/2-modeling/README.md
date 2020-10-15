# 2 - Modeling Pipeline

The modeling pipeline will:

- Update the training dataset with the latest version of data.
- Train and register the models.
- [Optional] Deploy the models into webservices ready to do real-time forecasting upon request.

## Instructions

Before creating the Azure DevOps pipeline:

1. Make sure the [AML extension](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml) is installed in the Azure DevOps organization.

2. Create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the Machine Learning Workspace that was created when you ran the [setup pipeline](../1-setup/). As you did [when you created a service connection before](../#1-create-service-connection), mark the option to grant access permission to all pipelines, and copy the name as you'll need it in the next step.

3. Modify the **``manymodels-vg``** [variable group you created before](../#2-create-variable-group), and add two new variables:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| SERVICECONNECTION_WORKSPACE | Name of the connection to the AML Workspace you have just created |
| TRAINING_METHOD             | "automl" or "customscript" depending which method you want to use for training the models |

Then, create the pipeline as you did in the [setup pipeline](../1-setup/), selecting branch **``v2-preview``** and setting the path to [`/mlops-pipelines/2-modeling/pipeline-modeling.yml`](pipeline-modeling.yml).

Run the pipeline and wait for it to finish.

## Result

The pipeline run should look like this (deployment stages will have run or not depending on your configuration, read more [here](Details.md#enabling-deployment)):

<img src="../../images/mlops_pipeline_2_modeling.png"
     width="1000"
     title="Modeling Pipeline"
     alt="Stages and jobs as described below" />

Containing the following stages and jobs:

- Update Data for Training
  - Download New Sample Files
  - Update Registered Training Dataset
- Run Model Training
  - Check Training Method
  - Publish Training AML Pipeline
  - Get Training Pipeline ID
  - Run Training
- Deploy Models to ACI [Optional]
  - Deploy Models
  - Register Routing Model [Optional]
  - Deploy Routing Webservice [Optional]
- Deploy Models to AKS [Optional]
  - Deploy Models
  - Register Routing Model [Optional]
  - Deploy Routing Webservice [Optional]

## Details

If you want to learn about the details and ways to customize the modeling pipeline please read the [details page](Details.md).
