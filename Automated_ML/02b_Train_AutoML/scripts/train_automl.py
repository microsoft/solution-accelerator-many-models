# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import argparse
import datetime
import hashlib
import json
import os
import tempfile
from multiprocessing import current_process
from pathlib import Path
from random import randint
from time import sleep

import pandas as pd
from azureml.automl.core.shared import constants
from azureml.automl.core.shared.exceptions import (AutoMLException,
                                                   ClientException, ErrorTypes)
from azureml.automl.core.shared.utilities import get_error_code
from azureml.core import Run
from azureml.core.model import Model
from azureml.train.automl import AutoMLConfig

from azureml_user.parallel_run import EntryScript
from train_automl_helper import compose_logs, str2bool

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
    entry_script = EntryScript()
    logger = entry_script.logger

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

    best_child_run, fitted_model = local_run.get_output()

    return fitted_model, local_run, best_child_run


def run(input_data):
    entry_script = EntryScript()
    logger = entry_script.logger
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    for file_path in input_data:
        current_run = None
        date1 = datetime.datetime.now()
        logger.info('start (' + file_path + ') ' + str(date1))

        file_name, file_extension = os.path.splitext(os.path.basename(file_path))

        result = {'ts_start': date1, 'file_name': file_name}

        try:
            if file_extension.lower() == ".parquet":
                data = pd.read_parquet(file_path)
            else:
                data = pd.read_csv(file_path, parse_dates=[timestamp_column])

            tags_dict = {'ModelType': 'AutoML'}
            for column_name in group_column_names:
                tags_dict.update({column_name: str(data[col_name].iloc[0])})
            result.update(tags_dict)

            if args.retrain_failed_models:
                logger.info('querying for existing models')
                try:
                    tags = [[k, v] for k, v in tags_dict.items()]
                    models = Model.list(current_step_run.experiment.workspace, tags=tags, latest=True)

                    if models:
                        logger.info("model already exists for the dataset " + models[0].name)
                        result['model_name'] = models[0].name
                        result['ts_end'] = datetime.datetime.now()
                        result_list.append(result)
                        continue
                except Exception as error:
                    logger.info('Failed to list the models. ' + 'Error message: ' + str(error))

            tags_dict.update({'InputData': file_path})
            tags_dict.update({'StepRunId': current_step_run.id})
            tags_dict.update({'RunId': current_step_run.parent.id})

            # train model
            fitted_model, current_run, best_child_run = train_model(file_path, data, logger)
            result['run'] = current_run.id

            model_string = '_'.join(str(v) for k, v in sorted(tags_dict.items()) if k in group_column_names).lower()
            logger.info("model string to encode " + model_string)
            sha = hashlib.sha256()
            sha.update(model_string.encode())
            model_name = 'automl_' + sha.hexdigest()
            tags_dict.update({'Hash': sha.hexdigest()})
            result['model_name'] = model_name
            try:
                logger.info('done training')
                print('Trained best model ' + model_name)

                logger.info(best_child_run)
                logger.info(fitted_model)
                logger.info(model_name)

                logger.info('register model')

                best_child_run.register_model(
                    model_name=model_name, model_path=constants.MODEL_PATH, description='AutoML', tags=tags_dict)

                print('Registered ' + model_name)
            except Exception as error:
                error_message = 'Failed to register the model. ' + 'Error message: ' + str(error)
                result.update({'error_type': ErrorTypes.Unclassified, 'error_message': error_message})
                from azureml.automl.core.shared import logging_utilities
                logging_utilities.log_traceback(error, None)
                logger.info(error_message)

        # 10.1 Log the error message if an exception occurs
        except (ClientException, AutoMLException) as error:
            error_message = 'Failed to train the model. Error ' + str(get_error_code(error)) + ': ' + str(error)
            logger.info(error_message)
            result['error_message'] = error_message
            result['error_code'] = get_error_code(error)
            if isinstance(error, AutoMLException):
                result['error_type'] = error.error_type

        date2 = datetime.datetime.now()
        result['ts_end'] = str(date2)
        result['status'] = current_run.get_status() if current_run else 'Failed'
        result_list.append(result)

        logger.info('ending (' + file_path + ') ' + str(date2))

    # Data returned by this function will be available in parallel_run_step.txt
    result = pd.DataFrame(result_list, columns=[
        *group_column_names, 'model_type', 'file_name', 'run_id', 'status', 
        'model_name', 'ts_start', 'ts_end', 'error_type', 'error_code', 'error_message'
    ])

    return result
