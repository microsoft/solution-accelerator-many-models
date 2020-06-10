# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Run
import pandas as pd
import numpy as np
import os
import argparse
import datetime
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from simple_lagger import SimpleLagger


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--forecast_granularity", type=int, help="frequency of forecasts daily weekly")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--model_type", type=str, help="input model type")

args, _ = parser.parse_known_args()

current_run = None

def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Loop through each file in the batch
        # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}
        start_datetime = datetime.datetime.now()

        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        data = pd.read_csv(csv_file_path, header=0)

        # 3.0 Create Features
        # Make a feature for day of the week and lagged values of the target up to order 4
        data['Week_Day'] = data[args.timestamp_column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').weekday())

        # Add a lag transform for making lagged features from the target
        # Make lags of orders 1 - 4
        lagger = SimpleLagger(args.target_column_name, args.time_column_name,
                              lag_orders=list(range(1, 4)))

        # For simplicity, drop the other features besides the day of week and lags  
        data = data.drop(['Price', 'Revenue', 'Store', 'Brand', 'Advert'], axis=1) 
        data = data.dropna()
        print(data)

        # 4.0 Prepare data for training
        X_train = data.drop(columns=[args.timestamp_column])
        y_train = X_train.pop(args.target_column)

        # 5.0 Make a Pipeline and train the model
        pipeline = Pipeline(steps=[('lagger', lagger), ('est', LinearRegression())])
        pipeline.fit(X_train, y_train)

        # 6.0 Save the modeling pipeline
        joblib.dump(pipeline, filename=os.path.join('./outputs/', model_name))

        # 7.0 Register the model to the workspace
        current_run.upload_file(model_name, os.path.join('./outputs/', model_name))

        tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType': args.model_type}
        current_run.register_model(model_path=model_name, model_name=model_name,
                                   model_framework=args.model_type, tags=tags_dict)

        # 8.0 Get in-sample predictions
        predictions = pipeline.predict(X_train)

        # 9.0 Calculate accuracy metrics
        mse = mean_squared_error(X_train[args.target_column], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(X_train[args.target_column], predictions)
        actuals = np.array(X_train[args.target_column])
        mape = np.mean(np.abs((actuals - predictions) / actuals) * 100)

        # 10.0 Log metrics
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)

        # 11.0 Add data to output
        end_datetime = datetime.datetime.now()
        result['store'] = store_name
        result['brand'] = brand_name
        result['model_type'] = args.model_type
        result['file_name'] = file_name
        result['model_name'] = model_name
        result['start_date'] = str(start_datetime)
        result['end_date'] = str(end_datetime)
        result['duration'] = str(end_datetime-start_datetime)
        result['mse'] = mse
        result['rmse'] = rmse
        result['mae'] = mae
        result['mape'] = mape
        result['index'] = idx
        result['num_models'] = len(input_data)
        result['status'] = current_run.get_status()

        print('ending (' + csv_file_path + ') ' + str(end_datetime))
        result_list.append(result)

    # Data returned by this function will be available in parallel_run_step.txt
    return pd.DataFrame(result_list)
