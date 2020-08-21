# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import datetime
import argparse

# Parse input arguments
parser = argparse.ArgumentParser("parallel run step results directory")
parser.add_argument("--parallel_run_step_output", type=str, help="output directory from parallel run step",
                    required=True)
parser.add_argument("--output_dir", type=str, help="output directory", required=True)
parser.add_argument("--target_column", type=str, help="column with actual values", default=None)
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data", required=True)
parser.add_argument("--timeseries_id_columns", type=str, nargs='*', required=True,
                    help="input columns identifying the timeseries")
# add list for the columns to pull ?

args, _ = parser.parse_known_args()

result_file = os.path.join(args.parallel_run_step_output, 'parallel_run_step.txt')

# Read the log file and set the column names from the input timeseries schema
# The parallel run step log does not have a header row, so add it for easier downstream processing
df_predictions = pd.read_csv(result_file, delimiter=" ", header=None)
pred_column_names = [args.timestamp_column, 'Prediction']
if args.target_column is not None:
    pred_column_names.append(args.target_column)
pred_column_names.extend(args.timeseries_id_columns)
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
