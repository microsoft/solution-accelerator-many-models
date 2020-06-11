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
from timeseries_utilities import SimpleLagger, SklearnForecaster


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

        # Read the data from CSV - parse timestamps as datetime type and put the time in the index
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column))

        # 3.0 Create Features
        # Make a feature for day of the week
        data['Week_Day'] = data.index.weekday.values

        # Drop columns that aren't helpful for modeling
        X_train = data.drop(columns=['Revenue', 'Store', 'Brand'])
        print(X_train.head())

        # 4.0 Make a Pipeline and train the model
        # Add a lag transform for making lagged features from the target
        # Make lags of orders 1 - 4
        lagger = SimpleLagger(args.target_column, args.timestamp_column,
                              lag_orders=list(range(1, 4)))

        # Wrap a linear regression model and make the pipeline
        estimator = SklearnForecaster(LinearRegression(), args.target_column)
        pipeline = Pipeline(steps=[('lagger', lagger), ('est', estimator)])
        pipeline.fit(X_train)

        # 5.0 Save the modeling pipeline
        joblib.dump(pipeline, filename=os.path.join('./outputs/', model_name))

        # 6.0 Register the model to the workspace
        current_run.upload_file(model_name, os.path.join('./outputs/', model_name))

        tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType': args.model_type}
        current_run.register_model(model_path=model_name, model_name=model_name,
                                   model_framework=args.model_type, tags=tags_dict)

        # 7.0 Get in-sample predictions and join with actuals for later comparison
        predictions = pipeline.predict(X_train)
        pred_column = 'predictions'
        compare_data = X_train.merge(predictions.to_frame(name=pred_column), how='left',
                                     left_index=True, right_index=True)
        compare_data.dropna(inplace=True)

        # 8.0 Calculate accuracy metrics
        mse = mean_squared_error(compare_data[args.target_column], compare_data[pred_column])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(compare_data[args.target_column], compare_data[pred_column])
        actuals = compare_data[args.target_column].values
        preds = compare_data[pred_column].values
        mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

        # 9.0 Log metrics
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)

        # 10.0 Add data to output
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
