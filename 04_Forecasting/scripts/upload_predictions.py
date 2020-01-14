import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Experiment, Datastore
import os
import datetime
from azureml.pipeline.core import PipelineRun
import argparse

# parse input arguments
parser = argparse.ArgumentParser("Upload prediction results arguments")

parser.add_argument("--parallelrunstep_name", type=str, help="input ParallelRunStep name")
parser.add_argument("--datastore", type=str, help="input datastore name")
parser.add_argument("--experiment", type=str, help="input experiment name")
parser.add_argument("--overwrite_predictions", type=str, help="input overwrite predictions True or False")
parser.add_argument("--pipeline_output_name", type=str, help="input ParralelRunStep output name")

args, _ = parser.parse_known_args()

print("Argument1 parallelrunstep_name: {}".format(args.parallelrunstep_name))
print("Argument2 datastore: {}".format(args.datastore))
print("Argument3 experiment: {}".format(args.experiment))
print("Argument4 overwrite_predictions: {}".format(args.overwrite_predictions))
print("Argument5 pipeline_output_name: {}".format(args.pipeline_output_name))

# set workspace and experiment
current_run = Run.get_context()
ws = current_run.experiment.workspace
experiment = Experiment(ws, args.experiment)

# retrieve the prediction file
pipeline_runId = current_run.get_details()['properties']['azureml.pipelinerunid']
pipeline_run = PipelineRun(experiment, pipeline_runId)
step_run = pipeline_run.find_step_run(args.parallelrunstep_name)[0]
prediction_output = step_run.get_output_data(args.pipeline_output_name)
prediction_output.download(local_path = "prediction")
print('Downloaded the prediction file of Pipeline Id: '+pipeline_runId)

# check the prediction file path
for root, dirs, files in os.walk("prediction"):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)
            print ('prediction file path: ' + result_file)

# read the file and clean up data
df_predictions = pd.read_csv(result_file, delimiter = " ", header = None)
df_predictions.columns = ['PredictedWeek','Store','Brand','PredictedQuantity']
print (df_predictions.head())
print ('Read and cleaned the prediction file')

# save the prediction file
output_path = os.path.join('./prediction/', 'forecasting_results')
df_predictions.to_csv(path_or_buf = output_path + '.csv', index = False)
print('Saved the forecasting_results.csv')

# upload the prediction file
forecast_dstore = Datastore(ws, args.datastore)
forecast_dstore.upload_files(['./prediction/forecasting_results'+'.csv'], target_path = 'oj_forecasts_'+str(datetime.datetime.now().date()), overwrite = args.overwrite_predictions, show_progress = True)
print('Uploaded the forecasting_results.csv')
