# Troubleshooting Tutorial

We developed customized log scripts to monitor training and scoring pipelines in a PowerBI dashboard.

In addition to the customized logging functionality, submitted pipelines will also generate azureml logs for each run.

When running into errors and existing logs are not suffice for troubleshooting, you can retrieve pipelines' azureml logs that contain more details. Here are 2 ways to retrieve the azureml logs.

## 1.0 How to retrieve azureml logs from Azure Machine Learning Studio

In Azure ML Studio, navigate to 'Pipelines' tab. Identify the failed pipeline run. Click on the failed step (e.g. 'many-models-training'). On the right hand side click on 'Logs'.

Under 'azureml-logs' tab, a file called '70_driver_log.txt' provides valuable information of the run. This log is a high level overview of the run.

In addition to driver log file, stderr_10.0.0.. files provide more details regarding the errors.

## 2.0 How to retrieve azureml logs from blob container

In the blob storage explorer, identify the default datastore and the subfolder named 'azureml'.

Each pipeline run will generate a unique Run ID and a folder named as the Run ID is created. Once you navigate to the desired Run ID folder, click on logs/sys/error. There should be folders named as 10.0.0.. and files inside the folders named as 'Process..'. These files are similar to the stderr_10.0.0.. files which provide details regarding the errors.
