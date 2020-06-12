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
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forecasting") #change this to tak the last date and start from there
parser.add_argument("--target_column", type=str, help="target colunm to predict on")
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data")
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
parser.add_argument("--model_type", type=str, help="model type")
parser.add_argument("--date_freq", type=str, help="the step size for predictions, daily, weekly")

args, _ = parser.parse_known_args()

current_run = None


def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up results dataframe
    results = pd.DataFrame()

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name

        # 3.0 Set up data to predict on
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column))

        # 4.0 Unpickle model and make predictions
        tag_list = [[id_col, data[id_col].iloc[0]] for id_col in args.timeseries_id_columns]
        tag_list.append(['ModelType', args.model_type])
        ws = Run.get_context().experiment.workspace
        models = Model.list(ws, tags=tag_list, latest=True)
        if len(models) > 1:
            raise ValueError("More than one models encountered for given timeseries id")
        model_path = models[0].download()
        forecaster = joblib.load(model_path)
        forecasts = forecaster.forecast(data)
        compare_data = data.assign(forecasts=forecasts).dropna()

        # 6.0 Calculate accuracy metrics for the forecast
        mse = mean_squared_error(compare_data[args.target_column], compare_data['forecasts'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(compare_data[args.target_column], compare_data['forecasts'])
        actuals = compare_data[args.target_column].values
        preds = compare_data['forecasts'].values
        mape = np.mean(np.abs((actuals - preds) / actuals) * 100)

        # 7.0 Log metrics
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)

        # 8.0 Add data to output
        end_datetime = datetime.datetime.now()
        result.update(ts_id_dict)
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
        

        prediction_df['Prediction'] = prediction_list
        prediction_df['Store'] = store_name
        prediction_df['Brand'] = brand_name
        prediction_df.drop(columns=['Week_Day'], inplace=True)
        

        results = results.append(prediction_df)

    return results
