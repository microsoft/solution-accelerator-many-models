# Troubleshooting

The Azure ML team provides guidance on [how to debug ParallelRunStep](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-parallel-run-step) as well as documentation on [debugging pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-pipelines#testing-scripts-locally). These are both valuable resources in debugging any issues with the solution accelerator.

## Logging

When running the pipelines, the quickest way to see if the pipeline is running successfully is to look at the logs. This can be done in [Azure ML Studio](https://ml.azure.com) or via your blob storage account.

While there's a lot of valuable information in logs, there's a couple of key files to look at first.

- ```70_driver_log.txt``` contains information from the controller that launches parallel run step code. This file will include any print statements that you put into ```train.py``` or ```forecast.py```.

- ```~/logs/sys/error/<ip_address>/Process-*.txt``` is the quickest ways to see errors in your pipeline. If the ```error``` folder doesn't exist, you likely haven't hit errors in your scripts yet. If it does, there's likely a problem.

- ```~/logs/sys/worker/<ip_address>/Process-*.txt``` provides detailed info about each mini-batch as it is picked up or completed by a worker. For each mini-batch, this file includes:
  - The IP address and the PID of the worker process.
  - The total number of items, successfully processed items count, and failed item count.
  - The start time, duration, process time and run method time.

## Known Issues

#### OJ Dataset download

The ```oj_sales_files.download(target_path, overwrite=True)``` may fail when downloading all 11,973 files. If you run into this issue, we recommend trying to download 5,000 or less files instead.


#### AutoML Dependencies are not guaranteed to be consistent with older versions of the SDK
When training your models using AutoML, all your runs may fail if you don't have the latest version of the  azureml-pipeline-steps package.

```
Incompatible/Missing packages found: azureml-automl-core requires azureml-dataprep<1.6.0a,>=1.4.10a but has azureml-dataprep 1.6.0.;azureml-defaults requires azureml-dataprep[fuse]<1.6.0a,>=1.4.10a but has azureml-dataprep 1.6.0.;azureml-automl-runtime requires azureml-dataprep[fuse,pandas]<1.6.0a,>=1.4.10a but has azureml-dataprep 1.6.0.;azureml-train-automl-runtime requires azureml-dataprep[fuse,pandas]<1.6.0a,>=1.4.10a but has azureml-dataprep 1.6.0.
```

This is a known issue and to ensure your runs are successful please get the latest versions of the notebook and make sure to updgrade the following packages:
Upgrade  the azureml-sdk[automl] and upgrade the azureml-pipeline-steps to a version greater than or equal to 1.6.0: 

```
pip install --upgrade azureml-sdk[automl]
pip install --upgrade azureml-pipeline-steps
```

#### 429 Errors

 When training or forecasting 10,000+ models, you may run into this error:

 ```
 RequestThrottled : Too Many requests: retry after 1 seconds
 ```

  or the folling error in the AML Studio UI:

  ```
  429 : Too Many requests: retry after 1 seconds
  ```

These errors are caused by UI frequently checking the log file contents and shouldn't affect the completion of the pipeline. You can check back once the pipeline finishes.
