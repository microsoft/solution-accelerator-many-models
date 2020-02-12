# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Run
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import argparse
import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from entry_script import EntryScript

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--stepwise_training", type=str, help="input stepwise training True or False")

args, _ = parser.parse_known_args()

current_run = None


def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up logging
    entry_script = EntryScript()
    logger = entry_script.logger
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Read in the data file
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = 'arima_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        data = pd.read_csv(csv_file_path, header = 0)
        logger.info(data.head())

        # 3.0 Split the data into train and test sets based on dates
        data = data.set_index(args.timestamp_column)
        max_date = datetime.datetime.strptime(data.index.max(), '%Y-%m-%d')
        split_date = max_date - timedelta(days=7 * args.n_test_periods)
        data.index = pd.to_datetime(data.index)
        train = data[data.index <= split_date]
        test = data[data.index > split_date]

        # 4.0 Train the model
        model = pm.auto_arima(train[args.target_column],
                start_p=0,
                start_q=0,
                test='adf', # default stationarity test is kpps
                max_p=3,
                max_d=2,
                max_q=3,
                m=3, # number of observations per seasonal cycle
                seasonal=True,
                information_criterion='aic',
                trace=True, # prints status on the fits
                stepwise=args.stepwise_training, # this increments instead of doing a grid search
                suppress_warnings=True,
                out_of_sample_size=16
        )
        model = model.fit(train[args.target_column])

        # 5.0 Save the model
        joblib.dump(model, filename=os.path.join('./outputs/', model_name))

        # 6.0 Register the model to the workspace
        current_run.upload_file(model_name, os.path.join('./outputs/', model_name))

        tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType': 'ARIMA'}
        current_run.register_model(model_path = model_name, model_name = model_name, model_framework = 'pmdarima',
                                   tags = tags_dict)

        # 7.0 Make predictions on test set
        prediction_list, conf_int = model.predict(args.n_test_periods, return_conf_int=True)

        # 9.0 Calculate accuracy metrics
        metrics = []
        mse = mean_squared_error(test['Quantity'], prediction_list)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['Quantity'], prediction_list)
        act, pred = np.array(test['Quantity']), np.array(prediction_list)
        mape = np.mean(np.abs((act - pred) / act) * 100)

        # 10.0 Log metrics
        current_run.log(model_name + '_aic', model.aic())
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)
        date2 = datetime.datetime.now()

        logger.info('ending (' + csv_file_path + ') ' + str(date2))

        result_list.append(True)
    return result_list
