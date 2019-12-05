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
In the real world, many problems can be too complex to be solved by a single model. Whether that be predicting sales for each individual store, building a predictive maintanence model for hundreds of oil wells, or tailoring an experience to individual users, building a model for each instance can lead to improved results on many machine learning problems.

Azure Machine Learning makes it easy to train, deploy, and manage hundreds or thousands of models. This repo will walk you through the end to end process of creating a many models solution from training to deploying to monitoring.

## Prerequisites

To use this solution accelerator, all you need is access to an Azure Subscription.

While it's not required, a basic understanding of Azure Machine Learning will be helpful for understanding the solution. The following resources can help introduce you to AML: adhfjh, ajdhfkj, ajdfhkjf.

## Getting started

Start by deploying the resources to Azure using button below. From there, walk through the folders chronologically following the steps outlined in each README.

<a href="https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FAzure-Samples%2Fazure-search-knowledge-mining%2Fmaster%2Fazuredeploy.json" target="_blank">
    <img src="http://azuredeploy.net/deploybutton.png"/>
</a>

## Contents




| Folder       | Description                                |
|-------------------|--------------------------------------------|
| `01_Data_Preparation`             | Sample source code.                        |
| `02_Training`      | Define what to ignore at commit time.      |
| `03_Scoring`    | List of changes to the sample.             |

## Key concepts



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
