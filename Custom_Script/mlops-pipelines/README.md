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

Create the pipeline as in [here](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline), selecting branch **``feature/mlops``** and setting the path to [/Custom_Script/mlops-pipelines/1-setup/setup-pipeline.yml](1-setup/setup-pipeline.yml).

## 2. Training Code Build Pipeline

The training code build pipeline will:
    - Create an Azure Machine Learning Pipeline that will train many models in parallel using the [train script](../scripts/train.py).
    - Publish the AML Pipeline into the AML workspace so it's ready to use whenever we want to retrain.

Before creating the Azure DevOps pipeline:

- Make sure the [AML extension](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml) is installed in the Azure DevOps organization.

- Create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the Machine Learning Workspace you created in the setup pipeline before. As you did before, mark the option to grant access permission to all pipelines, and copy the name as you'll need it in the next step.

- Modify the **``manymodels-vg``** variable group you created before, and add this new variable:  called **``manymodels-vg``**, 

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| SERVICECONNECTION_WORKSPACE | Name of the connection to the AML Workspace you have just created |

Then, create the pipeline as you did before, selecting branch **``feature/mlops``** and setting the path to [/Custom_Script/mlops-pipelines/2-training-code-build/training-code-build-pipeline.yml](2-training-code-build/training-code-build-pipeline.yml).

## 3. Modeling Pipeline

The modeling pipeline will:
    - Trigger the many models training by invoking the training AML Pipeline previously published.
    - Group the registered models according to specified tags.
    - Deploy each group into a different webservice hosted in ACI and/or AKS. These webservices will all use the same [forecast script](../scripts/forecast_webservice.py).
    - Deploy the entry point that will route the requests to the corresponding model webservice.

Create the pipeline as you did before, selecting branch **``feature/mlops``** and setting the path to [/Custom_Script/mlops-pipelines/3-modeling/modeling-pipeline.yml](3-modeling/modeling-pipeline.yml).
