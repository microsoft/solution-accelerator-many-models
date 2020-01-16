import pandas as pd
import os
import argparse
from sklearn.externals import joblib
from joblib import dump, load
import time
from datetime import timedelta
import logging
import datetime
from azureml.core.model import Model
from azureml.core import Experiment, Workspace, Run, Datastore
from entry_script_helper import EntryScriptHelper

# Get the information for the current Run
current_run = Run.get_context()

# Set the log file name
LOG_NAME = "user_log"

# Parse the arguments passed in the PipelineStep through the arguments option
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forcasting")

''' If you'd like to upload individual predictioon files as opposed to a concatenated prediction file,
    uncomment the following arguments and include these arguments in the ParralelRunStep. '''
# parser.add_argument("--output_datastore", type=str, help="input the name of registered forecast datastore")
# parser.add_argument("--overwrite_forecasting", type=str, help="True will over write the forecasting files")

args, _ = parser.parse_known_args()

print("Argument 1 forecast_horizon: {}".format(args.forecast_horizon))
print("Argument 2 starting_date: {}".format(args.starting_date))
# print("Argument 3(output_datastore): {}".format(args.output_datastore))
# print("Argument 4(overwrite_forecasting): {}".format(args.overwrite_forecasting))

def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")

def run(input_data):
    # 0. Set up Logging
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    all_predictions = pd.DataFrame()
    logger.info('making forecasts...')

    # 1. Iterate through the input data
    for idx, csv_file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = 'arima_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        logger.info('starting ('+csv_file_path+') ' + str(date1))
        current_run.log(model_name,'starttime-' + str(date1))

        # 2. Set up data to predict on
        store_list = [store_name] * args.forecast_horizon
        brand_list = [brand_name] * args.forecast_horizon
        date_list = pd.date_range(args.starting_date, periods = args.forecast_horizon, freq ='W-THU')

        prediction_df = pd.DataFrame(list(zip(date_list, store_list, brand_list)),
                                    columns = ['WeekStarting', 'Store', 'Brand'])

        # 3. Unpickle model and make predictions
        model_path = Model.get_model_path(model_name)
        model = joblib.load(model_path)
        print('Unpickled the model ' + model_name)

        prediction_list, conf_int = model.predict(args.forecast_horizon, return_conf_int = True)
        prediction_df['Predictions'] = prediction_list
        all_predictions = all_predictions.append(prediction_df)
        print('Made predictions ' + model_name)

        # 4. Save the output back to blob storage
        '''If you'd like to upload individual predictioon files as opposed to a concatenated prediction file,
        uncomment the following code block.'''
        #run_date = datetime.datetime.now().date()
        #ws = current_run.experiment.workspace
        #output_path = os.path.join('./outputs/', model_name + str(run_date))
        #prediction_df.to_csv(path_or_buf=output_path + '.csv', index = False)
        #forecasting_dstore = Datastore(ws, args.output_datastore)
        #forecasting_dstore.upload_files([output_path + '.csv'], target_path='oj_forecasts' + str(run_date),
        #                                overwrite=args.overwrite_forecasting, show_progress=True)

        # 5. Log the run
        date2 = datetime.datetime.now()
        logger.info('ending ('+str(csv_file_path)+') ' + str(date2))

    return all_predictions
