# 2 - Modeling Pipeline

The modeling pipeline will:

- Update the training dataset with the latest version of data.
- Train and register the models.
- [Optional] Deploy the models into webservices ready to do real-time forecasting upon request.

## Instructions

Before creating the Azure DevOps pipeline:

1. Make sure the [AML extension](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml) is installed in the Azure DevOps organization.

2. Create an **Azure Resource Manager** [service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) to access the Machine Learning Workspace you created in the setup pipeline before. As you did before, mark the option to grant access permission to all pipelines, and copy the name as you'll need it in the next step.

3. Modify the **``manymodels-vg``** [variable group you created before](../README.md/#2-create-variable-group), and add two new variables:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| SERVICECONNECTION_WORKSPACE | Name of the connection to the AML Workspace you have just created |
| TRAINING_METHOD             | "automl" or "customscript" depending which method you want to use for training the models |

Then, create the pipeline as you did before in the [setup pipeline](../1-setup/), selecting branch **``v2-preview``** and setting the path to [`/mlops-pipelines/2-modeling/pipeline-modeling.yml`](pipeline-modeling.yml).

## Result

The pipeline run should look like this:

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

### Data Update

The data update stage is there for demonstration purposes only, as the Orange Juice dataset is not going to change.
But in a real scenario the training dataset would be updated before training to incorporate the latest observations.

If you change the `DATASET_MAXFILES` variable in the [variable group](../README.md/#2-create-variable-group) after running the data preparation step in the previous pipeline, this step will update the training dataset with the new number of files. 

### Training

During the training stage, three main tasks are performed:

- Create an Azure Machine Learning Pipeline that will train many models in parallel and publish it into the AML workspace.
- Trigger the many models training by invoking the training AML Pipeline previously published.
- Register the models into the AML workspace.

The training will be performed using the `train.py` script in the corresponding [scripts folder](../../scripts/).
If you are using the Custom Script version, you should modify that script to meet your needs.

In both versions, AutoML and Custom Script, script settings are read from the `script_settings.json` file in the corresponding [configuration folder](../configuration/). These settings are right now based on the orange juice dataset. You can modify them if you want to use a different dataset.

#### Customizing tags in the model registry

When registered in the AML workspace, models are tagged with the model type ("lr" or "automl"), and the ID columns ("Store" and "Brand" in our example).

But if you have additional columns in you dataset that categorize you model (for example, a "City" column that applies to each of the stores you have), this can be specified in the [script settings](../configuration/README.md#customizing-script-settings) so this tag is also added to the model. To do this, we would add the following line to the `script_settings.json` file:

```
{
  ...
  "tag_columns": ["City"],
  ...
}
```

We do not have a "City" tag in our example dataset but have created three syntetic tags "StoreGroup10", "StoreGroup100" and "StoreGroup1000" that we use to showcase this functionality.

#### Dropping columns from dataset before training

Some columns in the dataset might be used for identifying or tagging the model, but shouldn't be considered in training. 
If that's the case, there are three different settings in the [settings file](../configuration/README.md#customizing-script-settings) you can use:

- `drop_id`: to drop the ID columns before training
- `drop_tags`: to drop the tag columns before training
- `drop_columns`: for any other additional column you want to drop before training

For example, setting these values to:

```
{
  ...
  "drop_id": true,
  "drop_tags": true,
  "drop_columns": ["Revenue"],
  ...
}
```

will drop all ID columns, all tag columns, and also the "Revenue" column.

#### Customize ParallelRunStep configuration

You can also modify the configuration for the ParallelRunStep, more details [here](../configuration/README.md#customizing-parallelrunstep-config).

### Deployment [Optional]

Three tasks are involved in the deployment stage:

- Group the registered models according to specified tags and maximum container size (500 by default).
- Deploy each group into a different webservice hosted in ACI and/or AKS. We call these "*model webservices*".
- [Optional] Deploy the entry point that will route the requests to the corresponding model webservice. We call this "*router webservice*".

All *model webservices* will use the same `model_webservice.py` script in the corresponding [scripts folder](../../scripts/).
The *router webservice* will use the `routing_webservice.py` script in the corresponding [scripts folder](../../scripts/).

#### Enabling deployment

Deployment of models for real-time forecasting is optional and disabled by default. 
To enable it, you must add a `DEPLOY_ACI` or/and `DEPLOY_AKS` variable to the [variable group](../README.md/#2-create-variable-group)
and set them to `true`. This will trigger the corresponding deployment stage.

Deploying in ACI (Azure Container Instances) is only recommended for development or testing pruposes, in a system in production you should use AKS (Azure Kubernetes Service) instead. In that case, you must also set variables to identify the AKS cluster you will be using.

These are the variables you have to set if you want to enable deployment of models:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| DEPLOY_ACI                  | Whether to deploy in ACI (`true`/`false`, default false) |
| DEPLOY_AKS                  | Whether to deploy in AKS (`true`/`false`, default false) |
| AKS_NAME                    | [Optional, only if DEPLOY_AKS is `true`] Name of the AKS resource you'll use for deploying the models |
| AKS_RESOURCE_GROUP          | [Optional, only if DEPLOY_AKS is `true`] Name of the resource group where the AKS resource is located |

When you enable AKS deployment, make sure you run the "Environment Setup" stage of the [setup pipeline](../1-setup/) before launching this pipeline, as the AKS cluster needs to be attached to the AML workspace first.

#### Changing the container size

By default, models are packed in groups of 500 and deployed in a single webservice container that will pick the correct model to generate predictions based on the information provided in the request body.

But you can change this behaviour by adding the following variable to the [variable group](../README.md/#2-create-variable-group):

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| MAX_CONTAINER_SIZE | Maximum number of models to fit into one webservice container |

Decreasing the maximum contaniner size will increase the number of webservices deployed, while increasing the size will make the models fit into less webservices. The maximum size supported for the moment is 1000.

#### Disabling the router

Since there is a limit in the number of models that can be deployed in the same webservice (see section above), when deploying many models we will most likely end up with several endpoints, each of them able to make predictions for different models.

By default, a router webservice is deployed to act as an entrypoint for all the requests, forward them to the appropiate model endpoint, and return the response back to the user. Sending a GET request to the router will return the mapping table it's using.

But you can disable the deployment of the router webservice by adding the following variable to the [variable group](../README.md/#2-create-variable-group):

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| DEPLOY_ROUTING_WEBSERVICE   | Set to `false` to avoid deploying the routing webservice  |

#### Customizing container configuration

The parameters used to configure the containers (AKS or ACI) are read from files placed in the corresponding [configuration/ folder](../configuration).

The files for the model container are:

- AKS: `model-deployment-config-aks.yml`
- ACI: `model-deployment-config-aci.yml`

The files for the router container are:

- AKS: `routing-deployment-config-aks.yml`
- ACI: `routing-deployment-config-aci.yml`

For more details on how to customize container configuration check [here](../configuration/README.md#customizing-parallelrunstep-config).

#### Customizing the update behaviour

In the third stage of the pipeline, models are deployed in a *smart way*, meaning that only models that have changed since last deployment are updated. New models are deployed, new versions are updated, models that have been deleted are removed, and the rest of them remain unchanged.

This means that if you change the deployment configuration file, these changes will only be applied in the webservices that contain new/updated models. If you want the new configuration to be applied to all webservices, you can do so by setting the `UPDATE_DEPLOYMENT` variable in the [variable group](../README.md/#2-create-variable-group).

If you want to resize the containers to allow a higher o lower number of models per webservice, you need to reset the deployment via the `RESET_DEPLOYMENT` variable. Please notice that webservices will not be available during this redeployment operation.

To sum up, the two variables that you can add to customize update behaviour are:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| RESET_DEPLOYMENT   | Set to `true` to reset existing webservices (to resize containers) |
| UPDATE_DEPLOYMENT  | Set to `true` to update all existing webservices (for config changes to apply) |

#### Customizing grouping

By default, models are packed in groups of fixed size following the order they have in the model registry.

If you want to **sort them before splitting in groups** of fixed size, you can set the order criteria in the `AML_MODEL_SORTING_TAGS` variable defined in the [`many-models-variables.yml`](../configuration/many-models-variables.yml) file.
For example, adding the following:

```
  - name: AML_MODEL_SORTING_TAGS
  - value: Store
```

would sort the models by store before splitting, ensuring that in most cases, the three models for each store (corresponding to the three different brands) would fall into the same webservice.

If you want to **split them by any particular criteria** you can specify that as well in the [`many-models-variables.yml`](../configuration/many-models-variables.yml) file, using the `AML_MODEL_SPLITTING_TAGS` variable. For example, if you had a "City" tag, adding the following:

```
  - name: AML_MODEL_SORTING_TAGS
  - value: City
```

would make each webservice contain models belonging to one specific city. We do not have a "City" tag in our example dataset but have created three syntetic tags "StoreGroup10", "StoreGroup100" and "StoreGroup1000" that group stores in groups of 10, 100 and 1000 that you can use if you want to test this functionality.
