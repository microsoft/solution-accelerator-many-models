# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import os
import tempfile
import uuid

from multiprocessing import current_process
from pathlib import Path
from azureml.core.dataset import Dataset
from azureml.core.model import Model
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import argparse
import hashlib
import pickle
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.shared import constants
import datetime
from entry_script import EntryScript
from train_automl_helper import str2bool, compose_logs
import logging
from azureml.automl.core.shared.exceptions import (AutoMLException,
                                                   ClientException, ErrorTypes)
from azureml.automl.core.shared.utilities import get_error_code


from joblib import dump, load
from random import randint
from time import sleep
import json


current_step_run = Run.get_context()

LOG_NAME = "user_log"


parser = argparse.ArgumentParser("split")
parser.add_argument("--process_count_per_node", default=1, type=int, help="number of processes per node")
parser.add_argument("--retrain_failed_models", default=False, type=str2bool, help="retrain failed models only")

args, _ = parser.parse_known_args()


def read_from_json():
    full_path = Path(__file__).absolute().parent
    with open(str(full_path) + "/automlconfig.json") as json_file:
        return json.load(json_file)


automl_settings = read_from_json()
# ''"{\"task\": \"forecasting\", \"iteration_timeout_minutes\": 10, \"iterations\": 10, \"n_cross_validations\": 3,
#  \"primary_metric\": \"accuracy\", \"preprocess\": false,  \"verbosity\": 20, \"label_column_name\": \"Quantity\",
#  \"debug_log\": \"automl_oj_sales_errors.log\", \"time_column_name\": \"WeekStarting\", \"max_horizon\": 6,
# \"drop_column_names\": [\"logQuantity\"], \"group_column_names\": [\"Store\", \"Brand\"]}"''

timestamp_column = automl_settings.get('time_column_name', None)
grain_column_names = automl_settings.get('grain_column_names', [])
group_column_names = automl_settings.get('group_column_names', [])
max_horizon = automl_settings.get('max_horizon', 0)
target_column = automl_settings.get('label_column_name', None)


print("max_horizon: {}".format(max_horizon))
print("target_column: {}".format(target_column))
print("timestamp_column: {}".format(timestamp_column))
print("group_column_names: {}".format(group_column_names))
print("grain_column_names: {}".format(grain_column_names))
print("retrain_failed_models: {}".format(args.retrain_failed_models))


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
        automl_settings['path'] = tempfile.mkdtemp()
        print(automl_settings['debug_log'])
        logger.info(f"{__file__}.AutoML debug log:{debug_log}")

    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")
    sleep(randint(1, 120))


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

    return fitted_model, local_run


def run(input_data):
    entry_script = EntryScript()
    logger = entry_script.logger
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    model_name = None
    current_run = None
    error_message = None
    error_code = None
    error_type = None
    tags_dict = None
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

            tags_dict = {'ModelType': 'AutoML'}

            for column_name in group_column_names:
                tags_dict.update({column_name: str(data.iat[0, data.columns.get_loc(column_name)])})

            if args.retrain_failed_models:
                logger.info('querying for existing models')
                try:
                    tags = [[k, v] for k, v in tags_dict.items()]
                    models = Model.list(current_step_run.experiment.workspace, tags=tags, latest=True)

                    if models:
                        logger.info("model already exists for the dataset " + models[0].name)
                        logs = compose_logs(file_name, models[0], date1)
                        resultList.append(logs)
                        continue
                except Exception as error:
                    logger.info('Failed to list the models. ' + 'Error message: ' + str(error))

            tags_dict.update({'InputData': file_path})
            tags_dict.update({'StepRunId': current_step_run.id})
            tags_dict.update({'RunId': current_step_run.parent.id})

            # train model
            fitted_model, current_run = train_model(file_path, data, logger)
            model_string = '_'.join(str(v) for k, v in sorted(tags_dict.items()) if k in group_column_names).lower()
            logger.info("model string to encode " + model_string)
            sha = hashlib.sha256()
            sha.update(model_string.encode())
            model_name = 'automl_' + sha.hexdigest()
            tags_dict.update({'Hash': sha.hexdigest()})
            try:
                logger.info('done training')
                print('Trained best model ' + model_name)

                logger.info(fitted_model)
                logger.info(model_name)

                logger.info('register model')

                current_run.register_model(model_name=model_name, description='AutoML', tags=tags_dict)
                print('Registered ' + model_name)
            except Exception as error:
                error_type = ErrorTypes.Unclassified
                error_message = 'Failed to register the model. ' + 'Error message: ' + str(error)
                logger.info(error_message)

            date2 = datetime.datetime.now()

            logs.append('AutoML')
            logs.append(file_name)
            logs.append(current_run.id)
            logs.append(current_run.get_status())
            logs.append(model_name)
            logs.append(tags_dict)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(error_type)
            logs.append(error_code)
            logs.append(error_message)

            logger.info('ending (' + file_path + ') ' + str(date2))

        # 10.1 Log the error message if an exception occurs
        except (ClientException, AutoMLException) as error:
            date2 = datetime.datetime.now()
            error_message = 'Failed to train the model. ' + 'Error : ' + str(error)

            logs.append('AutoML')
            logs.append(file_name)

            if current_run:
                logs.append(current_run.id)
                logs.append(current_run.get_status())
            else:
                logs.append(current_run)
                logs.append('Failed')

            logs.append(model_name)
            logs.append(tags_dict)
            logs.append(str(date1))
            logs.append(str(date2))
            if isinstance(error, AutoMLException):
                logs.append(error.error_type)
            else:
                logs.append(None)
            logs.append(get_error_code(error))
            logs.append(error_message)

            logger.info(error_message)
            logger.info('ending (' + file_path + ') ' + str(date2))

        resultList.append(logs)

    result = pd.DataFrame(data=resultList)
    return result
