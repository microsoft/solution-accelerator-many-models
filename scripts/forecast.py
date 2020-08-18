# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import os
import argparse
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from azureml.core.model import Model
from azureml.core.run import Run

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="target colunm to predict on", default=None)
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data", required=True)
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
parser.add_argument("--model_type", type=str, help="model type", required=True)

args, _ = parser.parse_known_args()

current_run = None


def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up results dataframe
    results = []

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name

        # 3.0 Set up data to predict on
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column))

        # 4.0 Unpickle model and make predictions
        ts_id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in args.timeseries_id_columns}
        tag_list = [list(kv) for kv in ts_id_dict.items()]
        tag_list.append(['ModelType', args.model_type])
        ws = Run.get_context().experiment.workspace
        models = Model.list(ws, tags=tag_list, latest=True)
        if len(models) > 1:
            raise ValueError("More than one models encountered for given timeseries id")
        model_path = models[0].download()
        forecaster = joblib.load(model_path)
        forecasts = forecaster.forecast(data)
        prediction_df = forecasts.to_frame(name='Prediction')

        # 5.0 If actuals are passed in with data, compute accuracy metrics and log in the Run
        # Also add actuals to the returned dataframe if they are available
        if args.target_column is not None and args.target_column in data.columns:
            compare_data = data.assign(forecasts=forecasts)
            mse = mean_squared_error(compare_data[args.target_column], compare_data['forecasts'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(compare_data[args.target_column], compare_data['forecasts'])
            actuals = compare_data[args.target_column].values
            preds = compare_data['forecasts'].values
            mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

            current_run.log(model_name + '_mse', mse)
            current_run.log(model_name + '_rmse', rmse)
            current_run.log(model_name + '_mae', mae)
            current_run.log(model_name + '_mape', mape)

            prediction_df[args.target_column] = data[args.target_column]

        # 6.0 Add the timeseries id columns and append the dataframe to the return list
        results.append(prediction_df.reset_index().assign(**ts_id_dict))

    # Data returned by this function will be available in parallel_run_step.txt
    output_columns = [*args.timeseries_id_columns, args.timestamp_column, 'Prediction']
    if args.target_column:
        output_columns.append(args.target_column)
    predictions = pd.concat(results)[output_columns]

    return predictions
