import pandas as pd
import os
import uuid

from azureml.core.dataset import Dataset
from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import argparse
import pickle
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig
from azureml.train.automl import AutoMLConfig
import azureml.automl.core
from automl.client.core.common import constants
import datetime
from entry_script_helper import EntryScriptHelper
from forecasting_helper import align_outputs
import logging

from sklearn.externals import joblib
from joblib import dump, load
import json

current_step_run = Run.get_context()
LOG_NAME = "user_log"

parser = argparse.ArgumentParser("split")

args, _ = parser.parse_known_args()


def read_from_json():
    with open('automlconfig.json') as json_file:
        return json.load(json_file)


automl_settings = read_from_json()
# ''"{\"task\": \"forecasting\", \"iteration_timeout_minutes\": 10, \"iterations\": 10, \"n_cross_validations\": 3,
#  \"primary_metric\": \"accuracy\", \"preprocess\": false,  \"verbosity\": 20, \"label_column_name\": \"Quantity\",
#  \"debug_log\": \"automl_oj_sales_errors.log\", \"time_column_name\": \"WeekStarting\", \"max_horizon\": 6,
# \"drop_column_names\": [\"logQuantity\"], \"group_column_names\": [\"Store\", \"Brand\"]}"''

timestamp_column = automl_settings.get('time_column_name', None)
grain_column_names = automl_settings.get('grain_column_names', [])
group_column_names = automl_settings.get('group_column_names', [])
n_test_periods = automl_settings.get('max_horizon', 0)
target_column = automl_settings.get('label_column_name', None)


print("n_test_periods: {}".format(n_test_periods))
print("target_column: {}".format(target_column))
print("timestamp_column: {}".format(timestamp_column))
print("group_column_names: {}".format(group_column_names))
print("grain_column_names: {}".format(grain_column_names))


def split_last_n_by_grain(df):
    """Group df by grain and split on last n rows for each group."""
    df_grouped = (df.sort_values(timestamp_column)  # Sort by ascending time
                  .groupby(grain_column_names, group_keys=False))
    print(df_grouped)
    df_head = df_grouped.apply(lambda dfg: dfg.iloc[:-n_test_periods])
    df_tail = df_grouped.apply(lambda dfg: dfg.iloc[-n_test_periods:])
    return df_head, df_tail


def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")


def train_model(data, file_name):
    print(data.head())
    if automl_settings['task'] == 'forecasting':
        train, test = split_last_n_by_grain(data)
    else:
        train, test = train_test_split(data, test_size=0.2, random_state=0)

    train.to_csv("./{}_train.csv".format(file_name), index=None, header=True)
    test.to_csv("./{}_test.csv".format(file_name), index=None, header=True)

    ws = current_step_run.experiment.workspace
    datastore = ws.get_default_datastore()
    datastore.upload_files(files=["./{}_train.csv".format(file_name), "./{}_test.csv".format(file_name)],
                           target_path='dataset/', overwrite=True, show_progress=True)

    train_dataset = Dataset.Tabular.from_delimited_files(path=datastore.path("dataset/{}_train.csv".format(file_name)))
    automl_config = AutoMLConfig(training_data=train_dataset,
                                 **automl_settings)

    local_run = current_step_run.submit_child(automl_config, show_output=True)
    print(local_run)
    best_run, fitted_model = local_run.get_output()

    u1 = uuid.uuid4()
    model_name = 'automl_' + str(u1)[0:16]
    return fitted_model, model_name, local_run


def test_model(fitted_model, file_name, model_name):
    ws = current_step_run.experiment.workspace
    datastore = ws.get_default_datastore()
    test_dataset = Dataset.Tabular.from_delimited_files(path=datastore.path("dataset/{}_test.csv".format(file_name)))
    test = test_dataset.to_pandas_dataframe()
    X_test = test
    y_test = X_test.pop(target_column).values
    print("test model head")
    print(X_test.head())
    y_predictions, X_trans = fitted_model.forecast(X_test)
    print('Made predictions on  ' + model_name)

    # Insert predictions to test set
    test['Predictions'] = y_predictions
    print(test.head())
    print('Inserted predictions ' + model_name)

    df_all = align_outputs(y_predictions, X_trans, X_test, y_test, target_column)

    scores = {}
    if automl_settings['task'] == 'forecasting':
        from azureml.automl.core._vendor.automl.client.core.common import metrics as mt
        scores = mt.compute_metrics_regression(df_all['predicted'], df_all[target_column],
                                               list(constants.Metric.SCALAR_REGRESSION_SET),
                                               None, None, None)

    return scores


def run(input_data):
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    logger.info('2')
    idx = 0
    for file in input_data:
        logs = []
        date1 = datetime.datetime.now()
        logger.info('start (' + file + ') ' + str(datetime.datetime.now()))
        data = pd.read_csv(file)
        csv_file_path = file

        file_name = csv_file_path.split('/')[-1][:-4]
        try:
            # train model
            fitted_model, model_name, current_run = train_model(data, file_name)
            logger.info('done training')
            print('Trained best model ' + model_name)

            logger.info(fitted_model)
            logger.info(model_name)

            logger.info('register model, skip the outputs prefix')

            tags_dict = {'ModelType': 'AutoML'}
            for column_name in group_column_names:
                tags_dict.update({column_name: str(data.iat[0, data.columns.get_loc(column_name)])})
            tags_dict.update({'InputData': csv_file_path})

            current_run.register_model(model_name=model_name, description='AutoML', tags=tags_dict)
            print('Registered ' + model_name)

            scores = test_model(fitted_model, file_name, model_name)
            print("[Test data scores]\n")

            for key, value in scores.items():
                print('{}:   {:.3f}'.format(key, value))
                logger.info('{}:   {:.3f}'.format(key, value))

            logger.info('Calculated metrics')

            date2 = datetime.datetime.now()

            logs.append('AutoML')
            logs.append(file_name)
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2 - date1))
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(current_run.get_status())
            idx += 1

            logger.info('ending (' + csv_file_path + ') ' + str(date2))

        # 10.1 Log the error message if an exception occurs
        except (ValueError, UnboundLocalError, NameError, ModuleNotFoundError, AttributeError, ImportError,
                FileNotFoundError, KeyError) as error:
            date2 = datetime.datetime.now()
            error_message = 'Failed to score the model. ' + 'Error message: ' + str(error)

            logs.append('AutoML')
            logs.append(file_name)
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2 - date1))
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(error_message)
            idx += 1

            logger.info('ending (' + csv_file_path + ') ' + str(date2))

        resultList.append(logs)

    return resultList
