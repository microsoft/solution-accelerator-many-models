# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import datetime
import argparse

# Parse input arguments
parser = argparse.ArgumentParser("parallel run step results directory")
parser.add_argument("--parallel_run_step_output", type=str, help="output directory from parallel run step")
parser.add_argument("--output_dir", type=str, help="output directory")
# add list for the columns to pull ?

args, _ = parser.parse_known_args()

result_file = os.path.join(args.parallel_run_step_output, 'parallel_run_step.txt')

# Read the log file and clean up data
df_predictions = pd.read_csv(result_file, delimiter=" ", header=None)
df_predictions.columns = ['Date', 'Store', 'Brand', 'PredictedQuantity']

# Save the log file
output_path = os.path.join(args.output_dir, 'forecasts_' + str(datetime.datetime.now().date()))
counter = 0
while os.path.exists(output_path + '.csv'):
    output_path += '_' + str(counter)
    counter += 1

df_predictions.to_csv(output_path + '.csv', index=False)
print('Saved the forecasting results to a csv')
