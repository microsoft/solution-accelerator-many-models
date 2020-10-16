# Many Models MLOps Pipelines

You'll use Azure DevOps for running the MLOps pipelines. Follow the steps below to set them up.

## 1. Create Many Models project in Azure DevOps

Create an [organization](https://docs.microsoft.com/azure/devops/organizations/accounts/create-organization?view=azure-devops#create-an-organization) and a [project](https://docs.microsoft.com/azure/devops/organizations/projects/create-project?view=azure-devops&tabs=preview-page#create-a-project) for the Many Models Solution Accelerator.

## 2. Create service connection

Next, create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the subscription and resource group where you plan to deploy the Many Models solution. The resource group should have already been created.

Mark the option to grant access permission to all pipelines. 

Choose any name you want and copy it, as you'll need it in the next step.

## 3. Create variable group

The last step you need to do before creating the pipelines is creating a variable group in Azure DevOps. Instructions on how to create a variable group are [here](https://docs.microsoft.com/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#create-a-variable-group).

Call it **``manymodels-vg``**, and add the following variables:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| RESOURCE_GROUP              | Name of the Azure Resource Group that you'll be using (should be already created) |
| SERVICECONNECTION_GROUP     | Name of the connection you created in the [last step](#1-create-service-connection) |
| LOCATION                    | [Azure location](https://azure.microsoft.com/global-infrastructure/locations/), no spaces |
| NAMESPACE                   | Unique naming prefix for created resources, to avoid name collisions |
| DATASET_MAXFILES            | Number of sample files to use (1 file = 1 model) |

## 3. Create pipelines

There are three pipelines in the Many Models Solution Accelerator:

1. [Setup Pipeline](1-setup/): set up workspace and data
2. [Modeling Pipeline](2-modeling/): training & optionally deploying all the models
3. [Batch Forecasting Pipeline](3-batch-forecasting/): generate batch predictions for all the models

Each pipeline is a YAML file that contains all the steps necessary to achieve each mision. The process for creating a pipeline from a YAML file in Azure DevOps is explained [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline).

Navigate into each pipeline's folder for specific instructions on how to set them up.

## [Optional] Customizing the pipelines

There are several things that can be customized in the Many Models Solution Accelerator. Please read the [details page](Details.md) if you want to find out more.
