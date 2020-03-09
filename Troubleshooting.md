# Troubleshooting

The Azure ML team provides guidance on [how to debug ParallelRunStep](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-parallel-run-step).

This [doc](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-pipelines#testing-scripts-locally)

## Logging

When running the pipelines, the quickest way to see if the script is running successfully is to look at the logs. This can be done in [Azure ML Studio](https://ml.azure.com) or via your blob stoage account.

#### 70_driver_log.txt

#### logs > sys > errors

## Known Issues

#### OJ Dataset download

The ```oj_sales_files.download(target_path, overwrite=True)``` may fail when downloading all 11,973 files. If you run into this issue, we recommend trying to download 5,000 or less files instead.

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
