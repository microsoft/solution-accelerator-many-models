import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Experiment, Datastore
import os
import datetime
from azureml.pipeline.core import PipelineRun
import argparse

# 1.0 parse input arguments
parser = argparse.ArgumentParser("Logging arguments")
parser.add_argument("--parallelrunstep_name", type=str, help="input ParralelRunStep name")
parser.add_argument("--datastore", type=str, help="input datastore name")
parser.add_argument("--experiment", type=str, help="input experiment name")
parser.add_argument("--overwrite_logs", type=str, help="input overwrite logs True or False")
parser.add_argument("--pipeline_output_name", type=str, help="input ParralelRunStep output name")

args, _ = parser.parse_known_args()

print("Argument1 parallelrunstep_name: {}".format(args.parallelrunstep_name))
print("Argument2 datastore: {}".format(args.datastore))
print("Argument3 experiment: {}".format(args.experiment))
print("Argument4 overwrite_logs: {}".format(args.overwrite_logs))
print("Argument5 pipeline_output_name: {}".format(args.pipeline_output_name))

# 2.0 set workspace and experiment
current_run = Run.get_context()
ws = current_run.experiment.workspace
experiment = Experiment(ws, args.experiment)

# 3.0 retrieve the log file
pipeline_runId = current_run.get_details()['properties']['azureml.pipelinerunid']
pipeline_run = PipelineRun(experiment, pipeline_runId)
step_run = pipeline_run.find_step_run(args.parallelrunstep_name)[0]
prediction_output = step_run.get_output_data(args.pipeline_output_name)
prediction_output.download(local_path = "logs")
print('Downloaded the log file of Pipeline Id: '+pipeline_runId)

# 4.0 check the log file path
for root, dirs, files in os.walk("logs"):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)
            print ('Log file path: ' + result_file)

# 5.0 read the file and clean up data
df_log = pd.read_csv(result_file, converters = {0: lambda x: x.strip("["),14: lambda x: x.strip("]")}, delimiter = ",", header = None)
df_log.columns = ['Store','Brand','ModelType','FileName','ModelName','StartTime','EndTime','Duration','MSE','RMSE','MAE','MAPE','Index','BatchSize','Status']
df_log['Store'] = df_log['Store'].apply(str).str.replace("'", '')
df_log['Brand'] = df_log['Brand'].apply(str).str.replace("'", '')
df_log['ModelType'] = df_log['ModelType'].apply(str).str.replace("'", '')
df_log['FileName'] = df_log['FileName'].apply(str).str.replace("'", '')
df_log['ModelName'] = df_log['ModelName'].apply(str).str.replace("'", '')
df_log['StartTime'] = df_log['StartTime'].apply(str).str.replace("'", '')
df_log['EndTime'] = df_log['EndTime'].apply(str).str.replace("'", '')
df_log['Duration'] = df_log['Duration'].apply(str).str.replace("'", '')
df_log['MSE'] = df_log['MSE'].apply(str).str.replace("'", '')
df_log['RMSE'] = df_log['RMSE'].apply(str).str.replace("'", '')
df_log['MAE'] = df_log['MAE'].apply(str).str.replace("'", '')
df_log['MAPE'] = df_log['MAPE'].apply(str).str.replace("'", '')
df_log['Status'] = df_log['Status'].apply(str).str.replace("'", '')
df_log['Status'] = df_log['Status'].apply(str).str.replace('"', '')
print (df_log.head())
print ('Read and cleaned the log file')

# 6.0 save the log file
output_path = os.path.join('./logs/', 'train_score_log')
df_log.to_csv(path_or_buf = output_path + '.csv', index = False)
print('Saved train_score_log.csv')

# 7.0 upload the log file
log_dstore = Datastore(ws, args.datastore)
log_dstore.upload_files(['./logs/train_score_log'+'.csv'], target_path = 'train_score_log_'+str(datetime.datetime.now().date()), overwrite = args.overwrite_logs, show_progress = True)
print('Uploaded train_score_log.csv')
