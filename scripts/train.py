# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import os
import argparse
import datetime
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from azureml.core import Run

from utils.timeseries import ColumnDropper, SimpleLagger, SimpleCalendarFeaturizer, SimpleForecaster


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, required=True, help="input target column")
parser.add_argument("--timestamp_column", type=str, required=True, help="input timestamp column")
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
parser.add_argument("--model_type", type=str, required=True, help="input model type")
parser.add_argument("--drop_columns", type=str, nargs='*', default=[],
                    help="list of columns to drop prior to modeling")

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
        start_datetime = datetime.datetime.now()
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name
        result = {
            'model_type': args.model_type, 'num_models': len(input_data), 'index': idx,
            'ts_start': start_datetime, 'file_name': file_name, 'model_name': model_name
        }

        # 1.0 Read the data from CSV - parse timestamps as datetime type and put the time in the index
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column))

        # 2.0 Create and fit the forecasting pipeline
        # The pipeline will drop unhelpful features, make a calendar feature, and make lag features
        lagger = SimpleLagger(args.target_column, lag_orders=[1, 2, 3, 4])
        transform_steps = [('column_dropper', ColumnDropper(args.drop_columns)),
                           ('calendar_featurizer', SimpleCalendarFeaturizer()), ('lagger', lagger)]
        forecaster = SimpleForecaster(transform_steps, LinearRegression(), args.target_column, args.timestamp_column)
        forecaster.fit(data)
        print('Featurized data example:')
        print(forecaster.transform(data).head())

        # 3.0 Save the forecasting pipeline
        joblib.dump(forecaster, filename=os.path.join('./outputs/', model_name))

        # 4.0 Register the model to the workspace
        try:
            current_run.upload_file(model_name, os.path.join('./outputs/', model_name))
            # Uses the values in the timeseries id columns from the first row of data to form tags for the model
            id_columns_dict = {id_col: str(data[id_col].iloc[0]) for id_col in args.timeseries_id_columns}
            result.update(id_columns_dict)
            tags_dict = {**id_columns_dict, 'ModelType': args.model_type}
            current_run.register_model(model_path=model_name, model_name=model_name,
                                       model_framework=args.model_type, tags=tags_dict)
        except Exception as error:
            error_message = 'Failed to register the model. Error message: ' + str(error)
            result.update({'error_type': 'Unclassified', 'error_message': error_message})
            print(error_message)

        end_datetime = datetime.datetime.now()
        result['ts_end'] = str(end_datetime)

        # 5.0 Get in-sample predictions and join with actuals
        forecasts = forecaster.forecast(data)
        compare_data = data.assign(forecasts=forecasts).dropna()

        # 6.0 Calculate accuracy metrics for the fit
        mse = mean_squared_error(compare_data[args.target_column], compare_data['forecasts'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(compare_data[args.target_column], compare_data['forecasts'])
        actuals = compare_data[args.target_column].values
        preds = compare_data['forecasts'].values
        mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

        # 7.0 Log metrics
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)
        result.update({'rmse': rmse, 'mae': mae, 'mape': mape})

        # 8.0 Add data to output
        result['status'] = current_run.get_status()
        result_list.append(result)

        print('ending (' + csv_file_path + ') ' + str(end_datetime))

    # Data returned by this function will be available in parallel_run_step.txt
    result = pd.DataFrame(result_list, columns=[
        *args.timeseries_id_columns, 'model_type', 'file_name', 'model_name', 'ts_start', 'ts_end',
        'rmse', 'mae', 'mape', 'index', 'num_models', 'status', 'error_type', 'error_message'
    ])

    return result
