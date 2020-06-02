# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import argparse
import joblib
from azureml.core.model import Model
from azureml.core.run import Run
import datetime

from utils.forecasting import format_prediction_data, update_prediction_data


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forecasting") #change this to tak the last date and start from there
parser.add_argument("--target_column", type=str, help="target colunm to predict on")
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data")
parser.add_argument("--model_type", type=str, help="model type")
parser.add_argument("--date_freq", type=str, help="the step size for predictions, daily, weekly")

args, _ = parser.parse_known_args()


def run(input_data):
    # 1.0 Set up results dataframe
    results = pd.DataFrame()

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        # 3.0 Set up data to predict on
        data = pd.read_csv(csv_file_path, header = 0)
        data[args.timestamp_column] = data[args.timestamp_column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

        prediction_df = format_prediction_data(
            data, args.forecast_horizon, args.date_freq,
            timestamp_column=args.timestamp_column,
            value_column=args.target_column
        )

        # 4.0 Unpickle model and make predictions
        ws = Run.get_context().experiment.workspace
        models = Model.list(ws, tags=[['Store', store_name], ['Brand', brand_name], ['ModelType', 'lr']], latest=True)
        if len(models) > 1:
            raise ValueError("More than one models encountered for store and brand")
        model_path = models[0].download()
        model = joblib.load(model_path)

        for i in range(len(prediction_df)):
            x_pred = prediction_df.loc[prediction_df.index == i].drop(columns=['Date', 'Prediction'])
            y_pred = model.predict(x_pred)[0]
            prediction_df = update_prediction_data(prediction_df, i, y_pred)

        prediction_df['Store'] = store_name
        prediction_df['Brand'] = brand_name
        
        results = results.append(prediction_df[['Date', 'Prediction', 'Store', 'Brand']])

    return results
