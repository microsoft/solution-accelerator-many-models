# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import joblib
import pandas as pd

from azureml.core.model import Model
from azureml.core.run import Run


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--id_columns", type=str, nargs='*', required=True, help="input columns identifying the model entity")
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data", required=True)
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
    for csv_file_path in input_data:
        
        # 3.0 Set up data to predict on
        data = (pd.read_csv(csv_file_path, parse_dates=[args.timestamp_column], header=0)
                .set_index(args.timestamp_column))

        # 4.0 Load registered model from Workspace
        id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in args.id_columns}
        tag_list = [list(kv) for kv in id_dict.items()]
        tag_list.append(['ModelType', args.model_type])
        ws = Run.get_context().experiment.workspace
        models = Model.list(ws, tags=tag_list, latest=True)
        if len(models) > 1:
            raise ValueError(f'More than one models encountered for given timeseries ID: {[m.name for m in models]}')
        model_path = models[0].download()
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
