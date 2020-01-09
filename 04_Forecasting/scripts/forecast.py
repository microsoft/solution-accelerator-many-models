
import pandas as pd
import os
import argparse
from sklearn.externals import joblib
from joblib import dump, load
import time
from datetime import timedelta
import logging 
import datetime

# Import the AzureML packages 
from azureml.core.model import Model
from azureml.core import Experiment, Workspace, Run, Datastore
from azureml.core import ScriptRunConfig
from azureml.core.run import Run

# Import the helper script 
from entry_script_helper import EntryScriptHelper


# Get the information for the current Run
current_run = Run.get_context()

# Set the log file name
LOG_NAME = "user_log"

# Parse the arguments passed in the PipelineStep through the arguments option 
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forcasting")
# parser.add_argument("--output_datastore", type=str, help="input the name of registered forecast datastore")
# parser.add_argument("--overwrite_forecasting", type=str, help="True will over write the forecasting files")

args, unknown = parser.parse_known_args()

print("Argument 1(forecast_horizon): %s" % args.forecast_horizon)
print("Argument 2(starting_date): %s" % args.starting_date)
# print("Argument 3(output_datastore): %s" % args.output_datastore)
# print("Argument 4(overwrite_forecasting): %s" % args.overwrite_forecasting)

def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")    
    return

def run(input_data):
    
    # 0. Set up Logging
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    all_predictions = pd.DataFrame()
    logger.info('making forecasts...')
    
    # 1. Iterate through the input data 
    for idx, file in enumerate(input_data):
        mname='arima'

        date1=datetime.datetime.now()
        logger.info('starting ('+file+') ' + str(date1))
        thisrun.log(mname,'starttime-'+str(date1))
        
        store = str(file).split('/')[-1][:-4].split('_')[0]
        brand = str(file).split('/')[-1][:-4].split('_')[-1]
        
        # 2. Set up data to predict on 
        store_list = [store] * args.forecast_horizon
        brand_list = [brand] * args.forecast_horizon
        date_list = pd.date_range(args.starting_date, periods = args.forecast_horizon, freq ='W-THU')
        
        prediction_df = pd.DataFrame(list(zip(date_list, store_list, brand_list)), 
                                    columns = ['WeekStarting', 'Store', 'Brand'])
        
        # 3. Unpickle Model and Make Predictions             
        model_name = 'arima_'+str(file).split('/')[-1][:-4]  
        model_path = Model.get_model_path(model_name)         
        model = joblib.load(model_path)        
        
        prediction_list, conf_int = model.predict(args.forecast_horizon, return_conf_int = True)

        prediction_df['Predictions'] = prediction_list
        
        all_predictions = all_predictions.append(prediction_df)

        # 4. Save the output back to blob storage 
        #run_date = datetime.datetime.now().date()
        #ws1 = thisrun.experiment.workspace
        #output_path = os.path.join('./outputs/', model_name + str(run_date))
        #prediction_df.to_csv(path_or_buf=output_path + '.csv', index = False)
        #forecasting_dstore = Datastore(ws1, args.output_datastore)
        #forecasting_dstore.upload_files([output_path + '.csv'], target_path='oj_forecasts' + str(run_date),
        #                                overwrite=args.overwrite_forecasting, show_progress=True)

        # 5. Log Metrics
        date2=datetime.datetime.now()
        logger.info('ending ('+str(file)+') ' + str(date2))     

    return all_predictions
