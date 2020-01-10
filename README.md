---
page_type: sample
languages:
- python
products:
- azureml
description: "Solution Accelerator designed to help get you up and running with the many models pattern on Azure"
urlFragment: "solution-accelerator-many-models"
---

# Many Models Solution Accelerator

<!-- 
Guidelines on README format: https://review.docs.microsoft.com/help/onboard/admin/samples/concepts/readme-template?branch=master

Guidance on onboarding samples to docs.microsoft.com/samples: https://review.docs.microsoft.com/help/onboard/admin/samples/process/onboarding?branch=master

Taxonomies for products and languages: https://review.docs.microsoft.com/new-hope/information-architecture/metadata/taxonomies?branch=master
-->
In the real world, many problems can be too complex to be solved by a single machine learning model. Whether that be predicting sales for each individual store, building a predictive maintanence model for hundreds of oil wells, or tailoring an experience to individual users, building a model for each instance can lead to improved results on many machine learning problems.

Azure Machine Learning makes it easy to train, operate, and manage hundreds or even thousands of models. This repo will walk you through the end to end process of creating a many models solution from training to scoring to monitoring.

## Prerequisites

To use this solution accelerator, all you need is access to an [Azure subscription](https://azure.microsoft.com/en-us/free/).

While it's not required, a basic understanding of Azure Machine Learning will be helpful for understanding the solution. The following resources can help introduce you to AML: [overview](https://azure.microsoft.com/en-us/services/machine-learning/), [tutorials](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup), [sample notebooks](https://github.com/Azure/MachineLearningNotebooks).

## Getting started

#### 1. Deploy Resources
Start by deploying the resources to Azure using button below: 

<a href="https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-search-knowledge-mining%2Fmaster%2Fazuredeploy.json" target="_blank">
    <img src="http://azuredeploy.net/deploybutton.png"/>
</a>

#### 2. Configure Development Environment
Next you'll need to configure your [development environment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment) for Azure Machine Learning. We recommend using a [cloud based compute instance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#compute-instance) as it's the fastest way to get up and running.

#### 3. Run Notebooks
From there, walk through the folders chronologically following the steps outlined in each notebook.

## Contents

In this repo, you'll train and score a forecasting model for each orange juice brand and for each store at a (simulated) grocery store. By the end, you'll have forecasted sales by using 11,973 models to predict sales for the next few weeks.

The data used in this sample is simulated based on the [Dominick's Orange Juice Dataset](http://www.cs.unitn.it/~taufer/QMMA/L10-OJ-Data.html#(1)), sales data from a Chicago area grocery store.

The functionality is broken into the following folders designed to be run step by step:
| Folder       | Description                                |
|-------------------|--------------------------------------------|
| `00_Environment_Setup`             | Configures your environment including deploying compute for training and configuring your blob storage for logging                         |
| `01_Data_Preparation`             | Walks through downloading the dataset from [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/catalog/) and registers the dataset with AML.                      |
| `02_Training`      | Creates a pipeline to train a model for each store and orange juice brand in the dataset.     |
| `03_Scoring`    | Creates a pipeline to score each of the models trained.           |
| `04_Forecasting`    | Creates a pipeline to forecast future orange juice sales.            |

## Key concepts

### ParallelRunStep
[ParallelRunStep](https://docs.microsoft.com/en-us/python/api/azureml-contrib-pipeline-steps/azureml.contrib.pipeline.steps.parallel_run_step.parallelrunstep?view=azure-ml-py) enables the parallel training of models.

### Pipelines
[Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines) allow you to create workflows in your machine learning projects. These workflows have a number of benefits including speed, simplicity, repeatability, and modularity.

### Other Concepts

In additional to ParallelRunStep and Pipelines, you'll also be working with the following concepts including [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace), [datasets](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data#datasets), [compute targets](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target#train), [python script steps](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.python_script_step.pythonscriptstep?view=azure-ml-py), and [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
