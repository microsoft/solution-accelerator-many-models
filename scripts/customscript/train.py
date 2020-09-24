# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import argparse
import datetime
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from azureml.core import Run

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from utils.models import get_model_name
from utils.timeseries import ColumnDropper, SimpleLagger, SimpleCalendarFeaturizer, SimpleForecaster


# 0.0 Parse input arguments and read settings
parser = argparse.ArgumentParser()
parser.add_argument("--settings-file", type=str, required=True, help="file containing the script settings")
args, _ = parser.parse_known_args()

base_path = Path(__file__).absolute().parent
with open(os.path.join(base_path, args.settings_file), 'r') as f:
    customscript_settings = json.load(f)

id_columns = customscript_settings['id_columns']
target_column = customscript_settings['target_column']
timestamp_column = customscript_settings['timestamp_column']
model_type = customscript_settings['model_type']
tag_columns = customscript_settings.get('tag_columns', [])
drop_id = customscript_settings.get('drop_id', False)
drop_tags = customscript_settings.get('drop_tags', False)
drop_columns = customscript_settings.get('drop_columns', [])
test_size = customscript_settings.get('test_size', 10)


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
        data = (pd.read_csv(csv_file_path, parse_dates=[timestamp_column], header=0)
                .set_index(timestamp_column)
                .sort_index(ascending=True))    

        # ID and tags: uses the values from the first row of data
        id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in id_columns}
        model_name = get_model_name(model_type, id_dict)
        tags_dict = {tag_col: str(data[tag_col].iloc[0]) for tag_col in tag_columns}
        cols_todrop = drop_columns + \
                      (id_columns if drop_id else []) + \
                      (tag_columns if drop_tags else [])
        print(f'Model name: "{model_name}"')
        print(f'ID tags: {id_dict}')
        print(f'Extra tags: {tags_dict}')
        print(f'Columns to drop: {cols_todrop}')

        # 2.0 Split the data into train and test sets
        train = data[:-test_size]
        test = data[-test_size:]

        # 3.0 Create and fit the forecasting pipeline
        # The pipeline will drop unhelpful features, make a calendar feature, and make lag features
        lagger = SimpleLagger(target_column, lag_orders=[1, 2, 3, 4])
        transform_steps = [('column_dropper', ColumnDropper(cols_todrop)),
                           ('calendar_featurizer', SimpleCalendarFeaturizer()), ('lagger', lagger)]
        forecaster = SimpleForecaster(transform_steps, LinearRegression(), target_column, timestamp_column)
        forecaster.fit(train)
        print('Featurized data example:')
        print(forecaster.transform(train).head())

        # 4.0 Get predictions on test set
        forecasts = forecaster.forecast(test)
        compare_data = test.assign(forecasts=forecasts).dropna()

        # 5.0 Calculate accuracy metrics for the fit
        mse = mean_squared_error(compare_data[target_column], compare_data['forecasts'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(compare_data[target_column], compare_data['forecasts'])
        actuals = compare_data[target_column].values
        preds = compare_data['forecasts'].values
        mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

        # 6.0 Log metrics
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)

        # 7.0 Train model with full dataset
        forecaster.fit(data)

        # 8.0 Save the forecasting pipeline and upload to workspace
        model_path = os.path.join(output_path, model_name)
        joblib.dump(forecaster, filename=model_path)
        current_run.upload_file(model_name, model_path)

        # 9.0 Register the model to the workspace to be used in forecasting
        all_tags = {'ModelType': model_type, **id_dict, **tags_dict}
        current_run.register_model(model_path=model_name, model_name=model_name,
                                   model_framework=model_type, tags=all_tags)

        # 10.0 Add data to output
        end_datetime = datetime.datetime.now()
        result.update(id_dict)
        result['model_type'] = model_type
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
