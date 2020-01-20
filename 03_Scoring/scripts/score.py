import pandas as pd
import os
import argparse
import numpy as np
import pmdarima as pm
from datetime import timedelta
import datetime
from entry_script import EntryScript
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
from azureml.core.model import Model
from azureml.core import Experiment, Workspace, Run, Datastore

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--n_test_periods", type=int, help="input number of predictions")
parser.add_argument("--timestamp_column", type=str, help="time column from the data")
parser.add_argument("--output_datastore", type=str, help="datastore to upload predictions to")
parser.add_argument("--overwrite_scoring", type=str, help="setting if the scoring files should be overwritten")

args, _ = parser.parse_known_args()

print("Argument 1 n_test_periods: {}".format(args.n_test_periods))
print("Argument 2 timestamp_column: {}".format(args.timestamp_column))
print("Argument 3 output_datastore: {}".format(args.output_datastore))
print("Argument 4 overwrite_scoring: {}".format(args.overwrite_scoring))

def run(input_data):
    # 1.0 Set up Logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Making predictions')
    current_run = Run.get_context()
    resultsList = []

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        logs = []

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = 'arima_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        logger.info('starting ('+csv_file_path+') ' + str(date1))

        # 3.0 Unpickle model and make predictions on test set
        try:
            model_path = Model.get_model_path(model_name)
            model = joblib.load(model_path)
            print('Unpickled ' + model_name)
            prediction_list, conf_int = model.predict(args.n_test_periods, return_conf_int = True)
            print('Made predictions on  ' + model_name)

            # 4.0 Split the data for test set and insert predictions
            data = pd.read_csv(csv_file_path, header=0)
            data = data.set_index(args.timestamp_column)
            max_date = datetime.datetime.strptime(data.index.max(), '%Y-%m-%d')
            split_date = max_date - timedelta(days = 7*args.n_test_periods)
            data.index = pd.to_datetime(data.index)
            test = data[data.index > split_date]

            test['Predictions'] = prediction_list
            print(test.head())

            # 5.0 Calculate accuracy metrics
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
            print('Calculated accuracy metrics  ' + model_name)
            print(metrics)

            # 5.1 Log accuracy metrics
            logger.info('accuracy metrics')
            logger.info(metrics)

            # Save scoring output back to blob storage as individual file (optional)
            '''
            If you want to return the predictions and acutal values for each model as a seperate file, use the code below to output the results
            of each iteration to the specified output_datastore.
            '''
            # run_date = datetime.datetime.now().date()
            # ws = current_run.experiment.workspace
            # output_path = os.path.join('./outputs/', model_name)
            # test.to_csv(path_or_buf=output_path + '.csv', index = False)

            # scoring_dstore = Datastore(ws, args.output_datastore)
            # scoring_dstore.upload_files([output_path +'.csv'], target_path = 'oj_scoring_' + str(run_date),
            #                             overwrite = args.overwrite_scoring, show_progress = True)

            # 6.0 Log the run
            date2 = datetime.datetime.now()

            logs.append(store_name)
            logs.append(brand_name)
            logs.append('ARIMA')
            logs.append(file_name)
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

            logger.info('ending ('+csv_file_path+') ' + str(date2))

        # 8.0 Log the error message if an exception occurs
        except (ValueError, UnboundLocalError, NameError, ModuleNotFoundError, AttributeError, ImportError, FileNotFoundError, KeyError) as error:
            date2 = datetime.datetime.now()
            error_message = 'Failed to score the model. ' + 'Error message: ' + str(error)

            logs.append(store_name)
            logs.append(brand_name)
            logs.append('ARIMA')
            logs.append(file_name)
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2-date1))
            logs.append('Null')
            logs.append('Null')
            logs.append('Null')
            logs.append('Null')
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(error_message)

            logger.info('ending ('+csv_file_path+') ' + str(date2))

    resultsList.append(logs)
    return resultsList
