# Instructions

You'll use Azure DevOps for running the MLOps pipelines. Create an [organization](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/create-organization?view=azure-devops#create-an-organization) an a [project](https://docs.microsoft.com/en-us/azure/devops/organizations/projects/create-project?view=azure-devops&tabs=preview-page#create-a-project) for the Many Models solution.

## 0. Before creating the pipelines

- Create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the subscription and resource group where you plan to deploy the Many Models solution. The resource group should already be created. Mark the option to grant access permission to all pipelines. Choose any name you want, and copy it as you'll need it in the next step.

- Create a [variable group](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#create-a-variable-group) called **``manymodels-vg``**, with the following variables:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| DATASET_MAXFILES            | Number of sample files to use (1 file = 1 model) |
| NAMESPACE                   | Unique naming prefix for created resources |
| LOCATION                    | [Azure location](https://azure.microsoft.com/en-us/global-infrastructure/locations/), no spaces |
| RESOURCE_GROUP              | Name of the Azure Resource Group that you'll be using (should be already created) |
| AKS_NAME                    | Name of the AKS resource you'll use for deploying the models |
| AKS_RESOURCE_GROUP          | Name of the resource group where the AKS resource is located |
| SERVICECONNECTION_GROUP     | Name of the connection you created in the last step |

## 1. Setup Pipeline

The setup pipeline will:
    - Deploy Azure Machine Learning and the other necessary resources into the resource group you specified.
    - Set up the Azure Machine Learning worskpace, creating a compute target and attaching the AKS cluster.
    - Download as many files as you specified in the DATASET_MAXFILES variable and register them as a dataset in AML.

Create the Pipeline as in [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline), selecting branch **``feature/mlops``** and setting the path to [/Custom_Script/mlops_pipelines/1-setup/setup-pipeline.yml](1-setup/setup-pipeline.yml).
