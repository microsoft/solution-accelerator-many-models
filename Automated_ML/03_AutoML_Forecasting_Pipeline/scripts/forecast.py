# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import argparse
from sklearn.externals import joblib
import datetime
import hashlib
from azureml.core import Run
from azureml.core.model import Model
from azureml_user.parallel_run import EntryScript


# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")


parser.add_argument("--group_column_names", '--nargs',
                    nargs='*', type=str, help="group_column_names")
parser.add_argument("--target_column_name", type=str,
                    help="target column", default=None)
parser.add_argument("--time_column_name", type=str,
                    help="time column", default=None)
parser.add_argument("--many_models_run_id",
                    type=str,
                    default=None,
                    required=False,
                    help="many_models_run_id: many models training run id.")

args, _ = parser.parse_known_args()

print("Argument 1 group_column_names: {}".format(args.group_column_names))
print("Argument 2 target_column_name: {}".format(args.target_column_name))
print("Argument 3 time_column_name: {}".format(args.time_column_name))
if hasattr(args, "many_models_run_id") and not args.many_models_run_id:
    print("Argument 4 many_models_run_id: {}".format(args.many_models_run_id))

current_step_run = Run.get_context()


def run(input_data):
    # 1.0 Set up Logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Making forecasts')
    os.makedirs('./outputs', exist_ok=True)

    all_predictions = pd.DataFrame()
    # 2.0 Iterate through input data
    for idx, file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        file_name, file_extension = os.path.splitext(
            os.path.basename(file_path))
        logger.info(file_path)
        if file_extension.lower() == ".parquet":
            data = pd.read_parquet(file_path)
        else:
            data = pd.read_csv(file_path)

        tags_dict = {}
        if hasattr(args, "many_models_run_id") and args.many_models_run_id:
            tags_dict['RunId'] = args.many_models_run_id

        for column_name in args.group_column_names:
            tags_dict.update(
                {column_name: str(data.iat[0, data.columns.get_loc(column_name)])})

        print(tags_dict)

        model_string = '_'.join(str(v) for k, v in sorted(
            tags_dict.items()) if k in args.group_column_names)
        logger.info("model string to encode " + model_string)
        sha = hashlib.sha256()
        sha.update(model_string.encode())
        model_name = 'automl_' + sha.hexdigest()

        logger.info('starting (' + file_path + ') ' + str(date1))

        ws = current_step_run.experiment.workspace
        logger.info('query the model ' + model_name)
        model_list = Model.list(ws, name=model_name,
                                tags=tags_dict, latest=True)

        if not model_list:
            print("Could not find model")
            continue

        # 4.0 Un-pickle model and make predictions
        model_path = model_list[0].download(exist_ok=True)
        model = joblib.load(model_path)
        model_name = model_list[0].name
        print('Unpickled the model ' + model_name)

        X_test = data.copy()
        if args.target_column_name is not None:
            X_test.pop(args.target_column_name)

        print("prediction data head")
        print(X_test.head())
        y_predictions, X_trans = model.forecast(
            X_test, ignore_data_errors=True)
        print('Made predictions ' + model_name)

        # Insert predictions to test set
        predicted_column_name = 'Predictions'
        data[predicted_column_name] = y_predictions
        print(data.head())
        print('Inserted predictions ' + model_name)

        cols = list(data.columns.values)
        print(cols)

        all_predictions = all_predictions.append(data)

        # 5.0 Log the run
        date2 = datetime.datetime.now()
        logger.info('ending (' + str(file_path) + ') ' + str(date2))

    print(all_predictions.head())
    return all_predictions
