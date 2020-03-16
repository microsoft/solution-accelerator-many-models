# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
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

args, _ = parser.parse_known_args()

print("Argument 3 group_column_names: {}".format(args.group_column_names))
print("Argument 3 target_column_name: {}".format(args.target_column_name))

RunContext = Run.get_context()


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
        print(tags)
        for k in tags_dict.keys():
            tags.append([k, tags_dict[k]])
        print(tags_dict[k])
        logger.info('starting (' + csv_file_path + ') ' + str(date1))

        ws = RunContext.experiment.workspace
        model_list = Model.list(ws, tags=tags, latest=True)

        if not model_list:
            print("Could not find model")
            continue

        # 4.0 Un-pickle model and make predictions
        model_path = model_list[0].download(exist_ok=True)
        model = joblib.load(model_path)
        model_name = model_list[0].name
        print('Unpickled the model ' + model_name)

        X_test = data
        if args.target_column_name is not None:
            X_test.pop(args.target_column_name)

        print("prediction data head")
        print(X_test.head())
        y_predictions, X_trans = model.forecast(X_test)
        print('Made predictions ' + model_name)

        # Insert predictions to test set
        X_test['Predictions'] = y_predictions
        print(X_test.head())
        print('Inserted predictions ' + model_name)

        all_predictions = all_predictions.append(X_test)

        # 5.0 Log the run
        date2 = datetime.datetime.now()
        logger.info('ending (' + str(csv_file_path) + ') ' + str(date2))

    return all_predictions
