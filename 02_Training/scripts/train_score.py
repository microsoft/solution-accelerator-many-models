from azureml.core import Experiment, Workspace, Run
from azureml.core.model import Model
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import argparse
import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
from entry_script import EntryScript

# 0.0 Parse input arguments
print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--stepwise_training", type=str, help="input stepwise training True or False")

args, _ = parser.parse_known_args()

print("Argument 1 n_test_periods: {}".format(args.n_test_periods))
print("Argument 2 target_column: {}".format(args.target_column))
print("Argument 3 timestamp_column: {}".format(args.timestamp_column))
print("Argument 4 stepwise_training: {}".format(args.stepwise_training))

def run(input_data):
    # 1.0 Set up logging
    entry_script = EntryScript()
    logger = entry_script.logger
    os.makedirs('./outputs', exist_ok=True)
    logger.info('Train and score models')
    current_run = Run.get_context()
    resultList = []

    # 2.0 Read in the data file
    for idx, csv_file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        logs = []
        logger.info('starting ('+csv_file_path+') ' + str(date1))

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = 'arima_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        data = pd.read_csv(csv_file_path, header = 0)
        logger.info(data.head())

        # 3.0 Split the data into train and test sets based on dates
        try:
            data = data.set_index(args.timestamp_column)
            max_date = datetime.datetime.strptime(data.index.max(),'%Y-%m-%d')
            split_date = max_date - timedelta(days = 7*args.n_test_periods)
            data.index = pd.to_datetime(data.index)
            train = data[data.index <= split_date]
            test = data[data.index > split_date]

            # 4.0 Train the model
            model = pm.auto_arima(train[args.target_column],
                    start_p = 0,
                    start_q = 0,
                    test = 'adf', #default stationarity test is kpps
                    max_p = 3,
                    max_d = 2,
                    max_q = 3,
                    m = 3, #number of observations per seasonal cycle
                    seasonal = True,
                    information_criterion = 'aic',
                    trace = True, #prints status on the fits
                    stepwise = args.stepwise_training, # this increments instead of doing a grid search
                    suppress_warnings = True,
                    out_of_sample_size = 16
                    )
            model = model.fit(train[args.target_column])
            logger.info('done training')
            print('Trained '+ model_name)

            # 5.0 Save the model
            logger.info(model)
            with open(model_name, 'wb') as file:
                joblib.dump(value = model, filename = os.path.join('./outputs/', model_name))
            print('Saved '+ model_name)

            # 6.0 Register the model to the workspace
            try:
                current_run.upload_file(model_name, os.path.join('./outputs/', model_name))
            except:
                logger.info('dont need to upload')
            logger.info('register model, skip the outputs prefix')

            tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType':'ARIMA'}
            current_run.register_model(model_path = model_name, model_name = model_name, model_framework = 'pmdarima', tags = tags_dict)
            print('Registered '+ model_name)

            # 7.0 Make predictions on test set
            prediction_list, conf_int = model.predict(args.n_test_periods, return_conf_int = True)
            print('Made predictions on  ' + model_name)

            # 8.0 Insert predictions to test set
            test['Predictions'] = prediction_list
            print(test.head())
            print('Inserted predictions ' + model_name)

            # 9.0 Calculate accuracy metrics
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

            # 9.1 Log accuracy metrics
            logger.info('accuracy metrics')
            logger.info(metrics)

            # 10.0 Log the run
            current_run.log(model_name + '_aic', model.aic())
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

        # 10.1 Log the error message if an exception occurs
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

    resultList.append(logs)
    return resultList
