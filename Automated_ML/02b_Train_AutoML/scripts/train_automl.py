# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import uuid

from multiprocessing import current_process
from pathlib import Path
from azureml.core.dataset import Dataset
from azureml.core.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import argparse
import pickle
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.shared import constants
import datetime
from entry_script_helper import EntryScriptHelper
import logging
from azureml.automl.core.shared.exceptions import (AutoMLException,
                                                   ClientException)


from sklearn.externals import joblib
from joblib import dump, load
import json

current_step_run = Run.get_context()

LOG_NAME = "user_log"

parser = argparse.ArgumentParser("split")
parser.add_argument("--process_count_per_node", default=1, type=int, help="number of processes per node")

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


def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    working_dir = os.environ.get("AZ_BATCHAI_OUTPUT_logs", "")
    ip_addr = os.environ.get("AZ_BATCHAI_WORKER_IP", "")
    log_dir = os.path.join(working_dir, "user", ip_addr, current_process().name)
    t_log_dir = Path(log_dir)
    t_log_dir.mkdir(parents=True, exist_ok=True)
    automl_settings['many_models'] = True
    automl_settings['many_models_process_count_per_node'] = args.process_count_per_node

    debug_log = automl_settings.get('debug_log', None)
    if debug_log is not None:
        automl_settings['debug_log'] = os.path.join(log_dir, debug_log)
        print(automl_settings['debug_log'])
        logger.info(f"{__file__}.AutoML debug log:{debug_log}")

    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")


def train_model(file_path, data, logger):
    file_name = file_path.split('/')[-1][:-4]
    print(file_name)
    logger.info("in train_model")
    print('data')
    print(data.head(5))
    automl_config = AutoMLConfig(training_data=data,
                                 **automl_settings)

    logger.info("submit_child")
    local_run = current_step_run.submit_child(automl_config, show_output=False)
    logger.info(local_run)
    print(local_run)
    local_run.wait_for_completion(show_output=True)

    fitted_model = local_run.get_output()

    u1 = uuid.uuid4()
    model_name = 'automl_' + str(u1)[0:16]
    return fitted_model, model_name, local_run


def run(input_data):
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    logger.info('2')
    idx = 0
    model_name = ''
    for file in input_data:
        logs = []
        date1 = datetime.datetime.now()
        logger.info('start (' + file + ') ' + str(datetime.datetime.now()))
        file_path = file

        file_name, file_extension = os.path.splitext(os.path.basename(file_path))

        try:
            if file_extension.lower() == ".parquet":
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path, parse_dates=[timestamp_column])
            # train model
            fitted_model, model_name, current_run = train_model(file_path, data, logger)

            try:
                logger.info('done training')
                print('Trained best model ' + model_name)

                logger.info(fitted_model)
                logger.info(model_name)

                logger.info('register model, skip the outputs prefix')

                tags_dict = {'ModelType': 'AutoML'}
                for column_name in group_column_names:
                    tags_dict.update({column_name: str(data.iat[0, data.columns.get_loc(column_name)])})
                tags_dict.update({'InputData': file_path})

                current_run.register_model(model_name=model_name, description='AutoML', tags=tags_dict)
                print('Registered ' + model_name)
            except Exception as error:
                error_message = 'Failed to register the model. ' + 'Error message: ' + str(error)
                logger.info(error_message)

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

            logger.info('ending (' + file_path + ') ' + str(date2))

        # 10.1 Log the error message if an exception occurs
        except (ValueError, UnboundLocalError, NameError, ModuleNotFoundError, AttributeError, ImportError,
                FileNotFoundError, KeyError, ClientException, AutoMLException) as error:
            date2 = datetime.datetime.now()
            error_message = 'Failed to train the model. ' + 'Error message: ' + str(error)

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

            logger.info(error_message)
            logger.info('ending (' + file_path + ') ' + str(date2))

        resultList.append(logs)

    return resultList
