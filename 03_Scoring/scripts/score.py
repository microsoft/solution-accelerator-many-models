

import pandas as pd
import os
import uuid
import argparse
import datetime
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
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig

# Import the helper script 
from entry_script_helper import EntryScriptHelper


# Get the information for the current Run
thisrun = Run.get_context()

# Set the log file name
LOG_NAME = "user_log"

# Parse the arguments passed in the PipelineStep through the arguments option 
parser = argparse.ArgumentParser("split")
parser.add_argument("--n_test_set", type=int, help="input number of predictions")
parser.add_argument("--timestamp_column", type=str, help="model name")

args, unknown = parser.parse_known_args()

print("Argument 1(n_test_set): %s" % args.n_test_set)
print("Argument 2(timestamp_column): %s" % args.timestamp_column)


def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")    
    return

def run(data):
    print("begin run ")
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    
    predictions = pd.DataFrame()
    
    logger.info('making predictions...')
    
    for file in data: 
    #for idx, file in enumerate(data): # add the enumerate for the 12,000 files 
        u1 = uuid.uuid4()
        mname='arima'+str(u1)[0:16]

        with thisrun.child_run(name=mname) as childrun:
            for w in range(0,5):
                thisrun.log(mname,str(w))
            
            date1=datetime.datetime.now()
            logger.info('starting ('+file+') ' + str(date1))
            childrun.log(mname,'starttime-'+str(date1))
            
            # 0. Unpickle Model 
            model_name = 'arima_'+str(data).split('/')[-1][:-4]  
            print(model_name)
            model_path = Model.get_model_path(model_name)         
            model = joblib.load(model_path)
            
            # 1. Make Predictions 
            prediction_list, conf_int = model.predict(args.n_test_set, return_conf_int = True)
            print("MAKING PREDICTIONS")
            
             
            # 2. Split the data for test set 
            data = pd.read_csv(file,header=0)
            data = data.set_index(args.timestamp_column)             
            max_date = datetime.datetime.strptime(data.index.max(),'%Y-%m-%d')
            split_date = max_date - timedelta(days=7*args.n_test_set)
            data.index = pd.to_datetime(data.index)
            test = data[data.index > split_date]
                
            test['Predictions'] = prediction_list
            print(test.head())
            
            # 3. Calculating Accuracy Metrics            
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
            ws1 = childrun.experiment.workspace
            output_path = os.path.join('./outputs/', model_name)
            test.to_csv(path_or_buf=output_path+'.csv', index = False)
            dstore = ws1.get_default_datastore()
            dstore.upload_files([output_path+'.csv'], target_path='oj_predictions', overwrite=False, show_progress=True)
            
            # 5. Append the predictions to return a dataframe if desired 
            predictions = predictions.append(test)
            
            # 6. Return metrics for logging
            date2=datetime.datetime.now()
            logger.info('ending ('+str(file)+') ' + str(date2))

            childrun.log(mname,'endtime-'+str(date2))
            childrun.log(mname,'auc-1')
        
    return predictions