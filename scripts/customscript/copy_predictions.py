# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import datetime
import argparse
from pathlib import Path
import pandas as pd

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--parallel_run_step_output", type=str, required=True, help="output directory from parallel run step")
parser.add_argument("--output_dir", type=str, required=True, help="output directory")
parser.add_argument("--settings-file", type=str, required=True, help="file containing the script settings")
args, _ = parser.parse_known_args()

base_path = Path(__file__).absolute().parent
with open(os.path.join(base_path, args.settings_file), 'r') as f:
    customscript_settings = json.load(f)

id_columns = customscript_settings['id_columns']
target_column = customscript_settings['target_column']
timestamp_column = customscript_settings['timestamp_column']

# Read the log file and set the column names from the input timeseries schema
# The parallel run step log does not have a header row, so add it for easier downstream processing
result_file = os.path.join(args.parallel_run_step_output, 'parallel_run_step.txt')
df_predictions = pd.read_csv(result_file, delimiter=" ", header=None)
pred_column_names = [timestamp_column, 'Prediction']
if target_column is not None:
    pred_column_names.append(target_column)
pred_column_names.extend(id_columns)
print('Using column names: {}'.format(pred_column_names))
assert len(df_predictions.columns) == len(pred_column_names), \
    'Number of columns in prediction data does not match given timeseries schema.'
df_predictions.columns = pred_column_names

# Save the log file
output_path = os.path.join(args.output_dir, 'forecasts_' + str(datetime.datetime.now().date()))
counter = 0
while os.path.exists(output_path + '.csv'):
    output_path += '_' + str(counter)
    counter += 1

df_predictions.to_csv(output_path + '.csv', index=False)
print('Saved the forecasting results to a csv')
