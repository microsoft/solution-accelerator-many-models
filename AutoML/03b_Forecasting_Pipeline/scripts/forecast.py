# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import os
import argparse
from sklearn.externals import joblib
from joblib import dump, load
import time
from datetime import timedelta
import datetime
from azureml.core.model import Model
from azureml.core import Experiment, Workspace, Run, Datastore
from azureml.train.automl import AutoMLConfig
import azureml.automl.core
from automl.client.core.common import constants
from entry_script import EntryScript

import json

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--group_column_names", '--nargs', nargs='*', type=str, help="group_column_names")
parser.add_argument("--target_column_name", type=str, help="target column", default=None)
parser.add_argument("--time_column_name", type=str, help="time column", default=None)

args, _ = parser.parse_known_args()

print("Argument 1 group_column_names: {}".format(args.group_column_names))
print("Argument 2 target_column_name: {}".format(args.target_column_name))
print("Argument 3 time_column_name: {}".format(args.time_column_name))

current_step_run = Run.get_context()


def run(input_data):
    # 1.0 Set up Logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Making forecasts')
    os.makedirs('./outputs', exist_ok=True)

    all_predictions = pd.DataFrame()
    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        data = pd.read_csv(csv_file_path)
        tags_dict = {}
        for column_name in args.group_column_names:
            tags_dict.update({column_name: str(data.iat[0, data.columns.get_loc(column_name)])})

        tags = [['ModelType', 'AutoML']]
        for k in tags_dict.keys():
            tags.append([k, tags_dict[k]])
        print(tags)
        logger.info('starting (' + csv_file_path + ') ' + str(date1))

        ws = current_step_run.experiment.workspace
        model_list = Model.list(ws, tags=tags, latest=True)

        if not model_list:
            print("Could not find model")
            continue

        # 4.0 Un-pickle model and make predictions
        model_path = model_list[0].download(exist_ok=True)
        model = joblib.load(model_path)
        model_name = model_list[0].name
        print('Unpickled the model ' + model_name)

        X_test = data.copy()
        y_test = None
        if args.target_column_name is not None:
            y_test = X_test.pop(args.target_column_name).values

        print("prediction data head")
        print(X_test.head())
        y_predictions, X_trans = model.forecast(X_test)
        print('Made predictions ' + model_name)

        # Insert predictions to test set
        predicted_column_name = 'Predictions'
        data[predicted_column_name] = y_predictions
        print(data.head())
        print('Inserted predictions ' + model_name)

        cols = list(data.columns.values)
        print(cols)

        if y_test is not None and args.time_column_name is not None:
            if X_test.dtypes[args.time_column_name] != 'datetime64[ns]':
                X_test[args.time_column_name] = pd.to_datetime(X_test[args.time_column_name])
            print("align_outputs")
            df_all = align_outputs(y_predictions, X_trans, X_test, y_test, args.target_column_name,
                                   predicted_column_name)
            print("rearrage columns so that predictions are last")
            df_all = df_all[cols]
            all_predictions = all_predictions.append(df_all)
        else:
            all_predictions = all_predictions.append(data)

        # 5.0 Log the run
        date2 = datetime.datetime.now()
        logger.info('ending (' + str(csv_file_path) + ') ' + str(date2))

    print(all_predictions.head())
    return all_predictions


def align_outputs(y_predicted, X_trans, X_test, y_test, target_column_name,
                  predicted_column_name='predicted',
                  horizon_colname='horizon_origin'):
    """
    Demonstrates how to get the output aligned to the inputs
    using pandas indexes. Helps understand what happened if
    the output's shape differs from the input shape, or if
    the data got re-sorted by time and grain during forecasting.

    Typical causes of misalignment are:
    * we predicted some periods that were missing in actuals -> drop from eval
    * model was asked to predict past max_horizon -> increase max horizon
    * data at start of X_test was needed for lags -> provide previous periods
    """

    if (horizon_colname in X_trans):
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted,
                                horizon_colname: X_trans[horizon_colname]})
    else:
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted})

    # y and X outputs are aligned by forecast() function contract
    df_fcst.index = X_trans.index

    # align original X_test to y_test
    X_test_full = X_test.copy()
    X_test_full[target_column_name] = y_test

    # X_test_full's index does not include origin, so reset for merge
    df_fcst.reset_index(inplace=True)
    X_test_full = X_test_full.reset_index().drop(columns='index')

    together = df_fcst.merge(X_test_full, how='right')

    # drop rows where prediction or actuals are nan
    # happens because of missing actuals
    # or at edges of time due to lags/rolling windows
    clean = together[together[[target_column_name,
                               predicted_column_name]].notnull().all(axis=1)]
    return(clean)
