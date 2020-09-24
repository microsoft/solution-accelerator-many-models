# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import joblib
import pandas as pd

from azureml.core.model import Model
from azureml.core.run import Run

from utils.models import get_model_name


# 0.0 Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--settings-file", type=str, required=True, help="file containing the script settings")
args, _ = parser.parse_known_args()

with open(args.settings_file, 'r') as f:
    customscript_settings = json.load(f)

id_columns = customscript_settings['id_columns']
timestamp_column = customscript_settings['timestamp_column']
model_type = customscript_settings['model_type']


def init():
    global ws
    current_run = Run.get_context()
    ws = current_run.experiment.workspace


def run(input_data):
    # 1.0 Set up results dataframe
    results = []

    # 2.0 Iterate through input data
    for csv_file_path in input_data:
        
        # 3.0 Set up data to predict on
        data = (pd.read_csv(csv_file_path, parse_dates=[timestamp_column], header=0)
                .set_index(timestamp_column))

        # 4.0 Load registered model from Workspace
        id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in id_columns}
        model_name = get_model_name(model_type, id_dict)
        model = Model(ws, model_name)
        model_path = model.download()
        forecaster = joblib.load(model_path)

        # 5.0 Make predictions
        forecasts = forecaster.forecast(data)
        prediction_df = forecasts.to_frame(name='Prediction')
        
        # 6.0 Add actuals to the returned dataframe if they are available
        if forecaster.target_column_name in data.columns:
            prediction_df[forecaster.target_column_name] = data[forecaster.target_column_name]
        
        # 7.0 Add the timeseries id columns and append the dataframe to the return list
        results.append(prediction_df.reset_index().assign(**id_dict))


    # Data returned by this function will be available in parallel_run_step.txt
    return pd.concat(results)
