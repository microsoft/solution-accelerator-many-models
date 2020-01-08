# Pipeline Tutorial

This README provides guidance on how to set ParallelRunConfig parameters and explains Entry Script. It's also applicable to scoring and forecasting pipeline.

## 1.0 How to set ParallelRunConfig parameters

In the [ParallelRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-contrib-pipeline-steps/azureml.contrib.pipeline.steps.parallel_run_config.parallelrunconfig), you will want to determine the number of workers and nodes appropriate for your use case. The workercount is based off the number of cores of the compute VM. The nodecount will determine the number of master nodes to use. In time-series ARIMA model scenario, increasing the node count will speed up the training process.


* <b>node_count</b>: The number of compute nodes to be used for running the user script. We recommend to start with 3 and increase the node_count if the training time is taking too long.

* <b>process_count_per_node</b>: Workercount - the number of processes per node. We are using a 8 cores computer cluster therefore we set it to 8.

* <b>compute_target</b>: Only AmlCompute is supported. You can change to a different compute cluster if one fails.

* <b>run_invocation_timeout</b>: The run() method invocation timeout in seconds. The timeout should be set to be higher than the maximum training time of one model (in seconds), by default it's 60. Since the model that takes the longest to train is about 120 seconds, we set it to be 500 which is greater than 120.   

* <b>entry_script</b>: The name of the training script.

* <b>source_directory</b>: Paths to folders that contain all files to execute on the compute target (optional).

* <b>environment</b>: The Python environment definition. You can configure it to use an existing Python environment or to set up a temporary environment for the experiment. The definition is also responsible for setting the required application dependencies (optional).

* <b>mini_batch_size</b>: The size of the mini-batch passed to a single run() call (optional).

    * For FileDataset, it's the number of files with a minimum value of 1. You can combine multiple files into one mini-batch. The default value is 1. In this orange juice sales example, we're using FileDataset and set mini_batch_size to be 1 because we're iterating through a list of FileDataset as our input data.

    * For TabularDataset, it's the size of data. Example values are 1024, 1024KB, 10MB, and 1GB. The recommended value is 1MB. The mini-batch from TabularDataset will never cross file boundaries. For example, if you have .csv files with various sizes, the smallest file is 100 KB and the largest is 10 MB. If you set mini_batch_size = 1MB, then files with a size smaller than 1 MB will be treated as one mini-batch. Files with a size larger than 1 MB will be split into multiple mini-batches.

* <b>error_threshold</b>: The number of record failures for TabularDataset and file failures for FileDataset that should be ignored during processing. If the error count for the entire input goes above this value, the job will be stopped. The error threshold is for the entire input and not for individual mini-batches sent to the run() method. The range is [-1, int.max]. The -1 part indicates ignoring all failures during processing. You can customize the error threshold based on your fault tolerance. Here we set it to 10, meaning that if 10 or more jobs failed, the job will be stopped and canceled.

* <b>output_action</b>: One of the following values indicates how the output will be organized -
    * <b>summary_only</b>: The user script will store the output. ParallelRunStep will use the output only for the error threshold calculation. The parallel_run_step.txt will return
    * <b>append_row</b>: For all input files, only one file will be created in the output folder to append all outputs separated by line. The file name will be parallel_run_step.txt. We set it to 'append_row' here because we collect the aggregated output file as our training log.


## 2.0 Entry Script

To train the models, you will need an entry script and a list of dependencies. The entry_script is a user script as a local file path that will be run in parallel on multiple nodes. If source_directly is present, use a relative path. Otherwise, use any path that's accessible on the machine.

The <b>entry script</b> accepts requests, tain and register the model, and then returns the results.

* <b>init()</b> - Typically this function loads the model into a global object. This function is run only once at the start of batch processing per worker node/process. init method can make use of following environment variables (ParallelRunStep input):

                AZUREML_BI_OUTPUT_PATH – output folder path

* <b>run(mini_batch)</b> - The method to be parallelized. Each invocation will have one minibatch.
    mini_batch: Batch inference will invoke run method and pass either a list or Pandas DataFrame as an argument to the method. Each entry in mini_batch will be - a filepath if input is a FileDataset, a Pandas DataFrame if input is a TabularDataset.

* <b>run method response</b>: run() method should return a Pandas DataFrame or an array. For append_row output_action, these returned elements are appended into the common output file. For summary_only, the contents of the elements are ignored. For all output actions, each returned output element indicates one successful inference of input element in the input mini-batch. User should make sure that enough data is included in inference result to map input to inference. Inference output will be written in output file and not guaranteed to be in order, user should use some key in the output to map it to input.
