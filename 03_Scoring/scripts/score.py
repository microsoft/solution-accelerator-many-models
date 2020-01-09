
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.externals import joblib
from joblib import dump, load
import pmdarima as pm
import time
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import logging

# Import the AzureML packages
from azureml.core.model import Model
from azureml.core import Experiment, Workspace, Run, Datastore
from azureml.core import ScriptRunConfig
from azureml.core.run import Run

# Import the helper script
from entry_script_helper import EntryScriptHelper

current_run = Run.get_context()
LOG_NAME = "user_log"

# Parse the arguments passed in the PipelineStep through the arguments option
parser = argparse.ArgumentParser("split")
parser.add_argument("--n_test_set", type=int, help="input number of predictions")
parser.add_argument("--timestamp_column", type=str, help="time column from the data")
parser.add_argument("--output_datastore", type=str, help="datastore to upload predictions to")
parser.add_argument("--overwrite_scoring", type=str, help="setting if the scoring files should be overwritten")

args, unknown = parser.parse_known_args()

print("Argument 1(n_test_set): %s" % args.n_test_set)
print("Argument 2(timestamp_column): %s" % args.timestamp_column)
print("Argument 3(output_datastore): %s" % args.output_datastore)
print("Argument 4(overwrite_scoring): %s" % args.overwrite_scoring)

def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")
    return

def run(input_data):
    print("begin run ")

    # 0. Set up Logging
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    resultsList = []
    predictions = pd.DataFrame()
    logger.info('making predictions...')

    for idx, file in enumerate(input_data): 
        mname='arima'
        print(mname)
        logs = []

        date1=datetime.datetime.now()
        logger.info('starting ('+file+') ' + str(date1))
        thisrun.log(mname,'starttime-'+str(date1))

        # 1. Unpickle Model and Make Predictions
        model_name = 'arima_'+str(file).split('/')[-1][:-4]
        print(model_name)
        model_path = Model.get_model_path(model_name)
        model = joblib.load(model_path)
        prediction_list, conf_int = model.predict(args.n_test_set, return_conf_int = True)

        # 2. Split the data for test set
        data = pd.read_csv(file, header=0)
        data = data.set_index(args.timestamp_column)
        max_date = datetime.datetime.strptime(data.index.max(), '%Y-%m-%d')
        split_date = max_date - timedelta(days = 7*args.n_test_set)
        data.index = pd.to_datetime(data.index)
        test = data[data.index > split_date]

        test['Predictions'] = prediction_list
        print(test.head())

        # 3. Calculate Accuracy Metrics
        metrics = []
        mse = mean_squared_error(test['Quantity'], test['Predictions'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['Quantity'], test['Predictions'])
        act, pred = np.array(test['Quantity']), np.array(test['Predictions'])
        mape = np.mean(np.abs((act - pred)/act)*100)

        metrics.append(mse)
        metrics.append(rmse)
        metrics.append(mae)
        metrics.append(mape)

        print(metrics)

        # add in a log for accuracy metrics
        logger.info('accuracy metrics')
        logger.info(metrics)

        # 4. Save the output back to blob storage
        '''
        If you want to return the predictions and acutal values for each model as a seperate file, use the code below to output the results 
        of each iteration to the specified output_datastore. 
        '''
#         run_date = datetime.datetime.now().date()
#         ws1 = thisrun.experiment.workspace
#         output_path = os.path.join('./outputs/', model_name)
#         test.to_csv(path_or_buf=output_path + '.csv', index = False)

#         scoring_dstore = Datastore(ws1, args.output_datastore)
#         scoring_dstore.upload_files([output_path +'.csv'], target_path = 'oj_scoring_' + str(run_date),
#                                     overwrite = args.overwrite_scoring, show_progress = True)

        # 5. Append the predictions to return a dataframe (optional)
        predictions = predictions.append(test)

        # 6. Log Metrics
        date2=datetime.datetime.now()
        logger.info('ending ('+str(file)+') ' + str(date2))
        
        store = str(file).split('/')[-1][:-4].split('_')[0]
        brand = str(file).split('/')[-1][:-4].split('_')[1]
        
        logs.append(store)
        logs.append(brand)
        logs.append('ARIMA')
        logs.append(str(file).split('/')[-1][:-4])
        logs.append(model_name)
        logs.append(str(date1))
        logs.append(str(date2))
        logs.append(str(date2-date1))
        logs.append(mse)
        logs.append(rmse)
        logs.append(mae)
        logs.append(mape)
        logs.append(idx)
        logs.append(len(input_data))
        logs.append(current_run.get_status())

    resultsList.append(logs)
    return resultsList
