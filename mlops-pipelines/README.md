# Instructions

You'll use Azure DevOps for running the MLOps pipelines. Create an [organization](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/create-organization?view=azure-devops#create-an-organization) an a [project](https://docs.microsoft.com/en-us/azure/devops/organizations/projects/create-project?view=azure-devops&tabs=preview-page#create-a-project) for the Many Models solution.

## 1. Create service connection

Next, create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the subscription and resource group where you plan to deploy the Many Models solution. The resource group should already be created. Mark the option to grant access permission to all pipelines. Choose any name you want, and copy it as you'll need it in the next step.

## 2. Create variable group

The last step you need to do before creating the pipeliens is creating a variable group in Azure DevOps. Instructions on how to create a variable group are [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#create-a-variable-group). 

Call it **``manymodels-vg``**, and add the following variables:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| RESOURCE_GROUP              | Name of the Azure Resource Group that you'll be using (should be already created) |
| SERVICECONNECTION_GROUP     | Name of the connection you created in the last step |
| LOCATION                    | [Azure location](https://azure.microsoft.com/en-us/global-infrastructure/locations/), no spaces |
| NAMESPACE                   | Unique naming prefix for created resources |
| DATASET_MAXFILES            | Number of sample files to use (1 file = 1 model) |

## 3. Create pipelines

There are three pipelines in the Many Models Solution Accelerator:

1. [Setup Pipeline](1-setup/): set up workspace and data
2. [Modeling Pipeline](2-modeling/): training & optionally deploying all the models
3. [Batch Forecasting Pipeline](3-batch-forecasting/): generate batch predictions for all the models

Each pipeline is a YAML file that contains all the steps necessary to achieve each mision. The process for creating a pipeline from a YAML file in Azure DevOps is explained [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline).

Navigate into each pipeline's folder for specific instructions on how to set them up. 

## Customizing the pipelines

There are several things that can be customized in the Many Models Solution Accelerator.

You may want to modify the training and forecasting scripts, specially if you are training using the Custom Script option.
All these scripts are located in [`scripts/`](../scripts/) folder in the repository root.

If you want to configure the compute targets, specify the specifics of your dataset or customize the names of the AML artifacts,
you should go to the [`configuration/`](configuration/) folder inside the MLOps Pipelines section.

If you want to make changes to the Azure resources deployed, check [`arm-templates/`](1-setup/arm-templates).
