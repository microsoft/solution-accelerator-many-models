
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Experiment, Datastore
import os
import datetime
from azureml.pipeline.core import PipelineRun
import argparse

# parse input arguments
parser = argparse.ArgumentParser("Logging arguments")

parser.add_argument("--ParallelRunStep_name", type=str, help="input ParralelRunStep name")
parser.add_argument("--datastore", type=str, help="input datastore name")
parser.add_argument("--experiment", type=str, help="input experiment name")
parser.add_argument("--overwrite_logs", type=str, help="input overwrite logs True or False")
parser.add_argument("--pipeline_output_name", type=str, help="input ParralelRunStep output name")

args, unknown = parser.parse_known_args()
print("Argument1 (ParallelRunStep_name): %s" % args.ParallelRunStep_name)
print("Argument2 (datastore): %s" % args.datastore)
print("Argument3 (experiment): %s" % args.experiment)
print("Argument4 (overwrite_logs): %s" % args.overwrite_logs)
print("Argument5 (pipeline_output_name): %s" % args.pipeline_output_name)

# set workspace and experiment
thisrun = Run.get_context()
ws = thisrun.experiment.workspace
experiment = Experiment(ws, args.experiment)

# retrieve the log file
pipeline_runId = thisrun.get_details()['properties']['azureml.pipelinerunid']
pipeline_run = PipelineRun(experiment, pipeline_runId)
step_run = pipeline_run.find_step_run(args.ParallelRunStep_name)[0]
prediction_output = step_run.get_output_data(args.pipeline_output_name)
prediction_output.download(local_path="logs")
print('Downloaded the log file of Pipeline Id: ' + pipeline_runId)

# check the log file path
for root, dirs, files in os.walk("logs"):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root, file)
            print('Log file path: ' + result_file)

# read the file and clean up data
df_log = pd.read_csv(result_file, converters={0: lambda x: x.strip("["), 10: lambda x: x.strip("]")},
                     delimiter=",", header=None)
df_log.columns = ['ModelType', 'FileName', 'ModelName', 'StartTime', 'EndTime', 'Duration', 'Index',
                  'BatchSize', 'Status']
df_log['ModelType'] = df_log['ModelType'].apply(str).str.replace("'", '')
df_log['FileName'] = df_log['FileName'].apply(str).str.replace("'", '')
df_log['ModelName'] = df_log['ModelName'].apply(str).str.replace("'", '')
df_log['StartTime'] = df_log['StartTime'].apply(str).str.replace("'", '')
df_log['EndTime'] = df_log['EndTime'].apply(str).str.replace("'", '')
df_log['Duration'] = df_log['Duration'].apply(str).str.replace("'", '')
df_log['Status'] = df_log['Status'].apply(str).str.replace("'", '')
print(df_log.head())
print('Read and cleaned the log file')

# save the log file
output_path = os.path.join('./logs/', 'training_log')
df_log.to_csv(path_or_buf=output_path + '.csv', index=False)
print('Saved the training_log.csv')

# upload the log file
log_dstore = Datastore(ws, args.datastore)
log_dstore.upload_files(['./logs/training_log' + '.csv'], target_path='training_log_' +
                        str(datetime.datetime.now().date()), overwrite=args.overwrite_logs, show_progress=True)
print('Uploaded the training_log.csv')
