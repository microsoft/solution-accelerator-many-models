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

from utils.models import get_model_name
from utils.timeseries import ColumnDropper, SimpleLagger, SimpleCalendarFeaturizer, SimpleForecaster


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--id_columns", type=str, nargs='*', required=True, help="input columns identifying the model entity")
parser.add_argument("--target_column", type=str, required=True, help="input target column")
parser.add_argument("--timestamp_column", type=str, required=True, help="input timestamp column")
parser.add_argument("--tag_columns", type=str, nargs='*', default=[], help="input columns to set as tags for the model")
parser.add_argument("--drop_id", action='store_true', help="flag to drop columns used as ID")
parser.add_argument("--drop_tags", action='store_true', help="flag to drop columns used as tags")
parser.add_argument("--drop_columns", type=str, nargs='*', default=[], help="list of columns to drop prior to modeling")
parser.add_argument("--model_type", type=str, required=True, help="input model type")
parser.add_argument("--test_size", type=int, required=True, help="number of observations to be used for testing")

args, _ = parser.parse_known_args()


def init():
    global current_run
    global output_path
    current_run = Run.get_context()
    output_path = './outputs/'
    os.makedirs(output_path, exist_ok=True)


def run(input_data):
    result_list = []

    # Loop through each file in the batch
    # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}
        start_datetime = datetime.datetime.now()

        # 1.0 Read the data from CSV - parse timestamps as datetime type and put the time in the index
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column)
                .sort_index(ascending=True))    

        # ID and tags: uses the values from the first row of data
        id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in args.id_columns}
        model_name = get_model_name(args.model_type, id_dict)
        tags_dict = {tag_col: str(data[tag_col].iloc[0]) for tag_col in args.tag_columns}
        cols_todrop = args.drop_columns + \
                      (args.id_columns if args.drop_id else []) + \
                      (args.tag_columns if args.drop_tags else [])
        print(f'Model name: "{model_name}"')
        print(f'ID tags: {id_dict}')
        print(f'Extra tags: {tags_dict}')
        print(f'Columns to drop: {cols_todrop}')

        # 2.0 Split the data into train and test sets
        train = data[:-args.test_size]
        test = data[-args.test_size:]

        # 3.0 Create and fit the forecasting pipeline
        # The pipeline will drop unhelpful features, make a calendar feature, and make lag features
        lagger = SimpleLagger(args.target_column, lag_orders=[1, 2, 3, 4])
        transform_steps = [('column_dropper', ColumnDropper(cols_todrop)),
                           ('calendar_featurizer', SimpleCalendarFeaturizer()), ('lagger', lagger)]
        forecaster = SimpleForecaster(transform_steps, LinearRegression(), args.target_column, args.timestamp_column)
        forecaster.fit(train)
        print('Featurized data example:')
        print(forecaster.transform(train).head())

        # 4.0 Get predictions on test set
        forecasts = forecaster.forecast(test)
        compare_data = test.assign(forecasts=forecasts).dropna()

        # 5.0 Calculate accuracy metrics for the fit
        mse = mean_squared_error(compare_data[args.target_column], compare_data['forecasts'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(compare_data[args.target_column], compare_data['forecasts'])
        actuals = compare_data[args.target_column].values
        preds = compare_data['forecasts'].values
        mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

        # 6.0 Log metrics
        current_run.log_row('rmse', **id_dict, value=rmse, model_name=model_name, **tags_dict)
        current_run.log_row('mae', **id_dict, value=mae, model_name=model_name, **tags_dict)
        current_run.log_row('mse', **id_dict, value=mse, model_name=model_name, **tags_dict)
        current_run.log_row('mape', **id_dict, value=mape, model_name=model_name, **tags_dict)

        # 7.0 Train model with full dataset
        forecaster.fit(data)

        # 8.0 Save the forecasting pipeline and upload to workspace
        model_path = os.path.join(output_path, model_name)
        joblib.dump(forecaster, filename=model_path)
        current_run.upload_file(model_name, model_path)

        # 9.0 Register the model to the workspace to be used in forecasting
        all_tags = {'ModelType': args.model_type, **id_dict, **tags_dict}
        current_run.register_model(model_path=model_name, model_name=model_name,
                                   model_framework=args.model_type, tags=all_tags)

        # 10.0 Add data to output
        end_datetime = datetime.datetime.now()
        result.update(id_dict)
        result['model_type'] = args.model_type
        result['file_name'] = csv_file_path
        result['model_name'] = model_name
        result['start_date'] = str(start_datetime)
        result['end_date'] = str(end_datetime)
        result['duration'] = str(end_datetime - start_datetime)
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
