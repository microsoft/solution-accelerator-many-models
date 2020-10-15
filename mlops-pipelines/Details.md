# Many Models MLOps Pipelines - Details

There are several things that can be customized in the Many Models Solution Accelerator.

### Modifying training / forecasting code

You may want to modify the training and forecasting scripts, specially if you are training using the Custom Script option.

All these scripts are located in [`scripts/`](../scripts/) folder in the repository root.

### Configure settings

If you want to configure the compute targets, specify the specifics of your dataset or customize the names of the AML artifacts,
you should go to the [`configuration/`](configuration/) folder inside the MLOps Pipelines section.

### Configure Azure resources

If you want to make changes to the Azure resources deployed, check [`arm-templates/`](1-setup/arm-templates).

### Variable group

There are many additional variables that can be added to the variable group. All of them are properly explained in the specific pipelines' folders, but below you can find a summary of all the variables supported:

| Variable Name               | Short description |
| --------------------------- | ----------------- |
| RESOURCE_GROUP              | Name of the Azure Resource Group that you'll be using |
| SERVICECONNECTION_GROUP     | Name of the connection to the resource group |
| LOCATION                    | [Azure location](https://azure.microsoft.com/global-infrastructure/locations/), no spaces |
| NAMESPACE                   | Unique naming prefix for created resources, to avoid name collisions |
| DATASET_MAXFILES            | Number of sample files to use (1 file = 1 model) |
| SERVICECONNECTION_WORKSPACE | Name of the connection to the AML Workspace |
| TRAINING_METHOD             | "automl" or "customscript" depending which method you want to use for training the models |
| DEPLOY_ACI                  | [Optional] Whether to deploy in ACI (`true`/`false`, default false) |
| DEPLOY_AKS                  | [Optional] Whether to deploy in AKS (`true`/`false`, default false) |
| AKS_NAME                    | [Optional, only if DEPLOY_AKS is `true`] Name of the AKS resource you'll use for deploying the models |
| AKS_RESOURCE_GROUP          | [Optional, only if DEPLOY_AKS is `true`] Name of the resource group where the AKS resource is located |
| MAX_CONTAINER_SIZE | [Optional] Maximum number of models to fit into one webservice container (default 500) |
| DEPLOY_ROUTING_WEBSERVICE   | [Optional] Set to `false` to avoid deploying the routing webservice |
| RESET_DEPLOYMENT   | [Optional] Set to `true` to reset existing webservices (to resize containers) |
| UPDATE_DEPLOYMENT  | [Optional] Set to `true` to update all existing webservices (for config changes to apply) |
