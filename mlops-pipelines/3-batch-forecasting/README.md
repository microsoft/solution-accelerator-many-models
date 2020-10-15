# 3 - Batch Forecasting Pipeline

The batch forecasting pipeline will:

- Update the inference dataset with the latest version of data.
- Launch batch forecasting and create batch predictions for all the models.

## Instructions

Just create the pipeline as you did before in the [setup pipeline](../1-setup/) and [modeling pipeline](../2-modeling/), selecting branch **``v2-preview``** and setting the path to [`/mlops-pipelines/3-batch-forecasting/pipeline-batch-forecasting.yml`](pipeline-batch-forecasting.yml).

## Result

The pipeline run should look like this:

<img src="../../images/mlops_pipeline_3_batchforecasting.png"
     width="1000"
     title="Batch Forecasting Pipeline"
     alt="Stages and jobs as described below" />

Containing the following stages and jobs:

- Update Data for Batch Forecasting
  - Download New Sample Files
  - Update Registered Inference Dataset
- Run Model Forecasting
  - Check Training Method
  - Publish Forecasting AML Pipeline
  - Get Forecasting Pipeline ID
  - Run Forecasting

## Details

### Data Update

The data update stage is there for demonstration purposes only, as the Orange Juice dataset is not going to change.
But in a real scenario the dataset would be updated before launching batch forecasting to incorporate the latest observations and make predictions into the future.

If you change the `DATASET_MAXFILES` variable in the [variable group](../README.md/#2-create-variable-group) after running the data preparation step in the previous pipeline, this step will update the inference dataset with the new number of files, but will fail to make forecasts for some models if no training has been run for these new files.

### Batch Forecasting

During the forecasting stage, three main tasks are performed:

- Create an Azure Machine Learning Pipeline that will generate batch forecasts for all the models in parallel, and publish it into the AML workspace.
- Trigger the many models batch forecasting by invoking the batch forecasting AML Pipeline previously published.
- Store the predictions in a separate container.

The predictions will be generated using the `forecast.py` script in the corresponding [scripts folder](../../scripts/).
If you are using the Custom Script version, you should modify that script to meet your needs.

In both versions, AutoML and Custom Script, script settings are read from the `script_settings.json` file in the corresponding [configuration folder](../configuration/). These settings are right now based on the orange juice dataset. You can modify them if you want to use a different dataset.

#### Customizing name of the predictions container

The name of the container where the predictions will be stored is defined in the [`many-models-variables.yml`](../configuration/many-models-variables.yml) file and it's set to "predictions" right now:

```
  - name: PREDICTIONS_CONTAINER_NAME
    value: predictions
```

You can modify that file if you want a different name for that container.

#### Customize ParallelRunStep configuration

You can also modify the configuration for the ParallelRunStep, more details [here](../configuration/README.md#customizing-parallelrunstep-config).
