# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from train_helper import MetadataFileHandler

# TODO: Remove once Batch AI has fixed this issue.
# Exclude mounted blobfuse folders from sys.path, preventing Python from scanning
# folders in the blob container when resolving import statements. This significantly reduces traffic
# to the storage account.
import sys
sys.path = [p for p in sys.path if not p.startswith('/mnt/batch')]

import datetime  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402
from azureml.automl.core.shared import constants  # noqa: E402
from azureml.automl.core.shared.exceptions import AutoMLException, ClientException, ErrorTypes  # noqa: E402
from azureml.automl.core.shared.utilities import get_error_code  # noqa: E402
from azureml.core import Run  # noqa: E402
from azureml.core.model import Model  # noqa: E402
from azureml.train.automl import AutoMLConfig  # noqa: E402


# This is used by UI to display the many model settings
many_model_run_properties = {'many_models_run': True}


def compose_logs(file_name, model, start_time):
    logs = []
    logs.append('AutoML')
    logs.append(file_name)
    logs.append(None)
    logs.append(None)
    logs.append(model.name)
    logs.append(model.tags)
    logs.append(start_time)
    logs.append(datetime.datetime.now())
    logs.append(None)
    logs.append(None)
    logs.append(None)
    return logs


def train_model(file_path, data, automl_settings, current_step_run):
    file_name = file_path.split('/')[-1][:-4]
    print(file_name)
    print("in train_model")
    print('data')
    print(data.head(5))
    print(automl_settings)
    automl_config = AutoMLConfig(training_data=data, **automl_settings)

    print("submit_child")
    local_run = current_step_run.submit_child(automl_config, show_output=True)

    local_run.add_properties({
        k: str(many_model_run_properties[k])
        for k in many_model_run_properties
    })

    print(local_run)

    best_child_run, fitted_model = local_run.get_output()

    return fitted_model, local_run, best_child_run


def run(file_path, args, automl_settings, current_step_run):
    timestamp_column = automl_settings.get('time_column_name', None)
    group_column_names = automl_settings.get('group_column_names', [])

    model_name = None
    current_run = None
    error_message = None
    error_code = None
    error_type = None
    tags_dict = None

    logs = []
    date1 = datetime.datetime.now()
    print('start (' + file_path + ') ' + str(datetime.datetime.now()))

    file_name_with_extension = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    try:
        if file_extension.lower() == ".parquet":
            data = pd.read_parquet(file_path)
        else:
            data = pd.read_csv(file_path, parse_dates=[timestamp_column])

        tags_dict = {'ModelType': 'AutoML'}
        group_columns_dict = {}

        for column_name in group_column_names:
            group_columns_dict.update({column_name: str(data.iat[0, data.columns.get_loc(column_name)])})

        tags_dict.update(group_columns_dict)

        if args.retrain_failed_models:
            print('querying for existing models')
            try:
                tags = [[k, v] for k, v in tags_dict.items()]
                models = Model.list(current_step_run.experiment.workspace, tags=tags, latest=True)

                if models:
                    print("model already exists for the dataset " + models[0].name)
                    return compose_logs(file_name, models[0], date1)
            except Exception as error:
                print('Failed to list the models. ' + 'Error message: ' + str(error))

        tags_dict.update({'InputData': file_name_with_extension})
        tags_dict.update({'StepRunId': current_step_run.id})
        tags_dict.update({'RunId': current_step_run.parent.id})

        # train model
        many_model_run_properties['many_models_data_tags'] = group_columns_dict
        many_model_run_properties['many_models_input_file'] = file_name_with_extension

        fitted_model, current_run, best_child_run = train_model(file_path, data, automl_settings, current_step_run)
        model_string = '_'.join(str(v) for k, v in sorted(group_columns_dict.items()))
        print("model string to encode " + model_string)
        sha = hashlib.sha256()
        sha.update(model_string.encode())
        model_name = 'automl_' + sha.hexdigest()
        tags_dict.update({'Hash': sha.hexdigest()})
        try:
            print('done training')
            print('Trained best model ' + model_name)

            print(best_child_run)
            print(fitted_model)
            print(model_name)

            print('register model')

            best_child_run.register_model(
                model_name=model_name, model_path=constants.MODEL_PATH, description='AutoML', tags=tags_dict)
            print('Registered ' + model_name)
        except Exception as error:
            error_type = ErrorTypes.Unclassified
            error_message = 'Failed to register the model. ' + 'Error message: ' + str(error)
            print(error_message)

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

        print('ending (' + file_path + ') ' + str(date2))

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

        print(error_message)
        print('ending (' + file_path + ') ' + str(date2))

    return logs


if __name__ == '__main__':
    data_file_path = sys.argv[1]
    data_dir = sys.argv[2]
    metadata_file_handler = MetadataFileHandler(data_dir)
    args = metadata_file_handler.load_args()
    automl_settings = metadata_file_handler.load_automl_settings()
    run_dto = metadata_file_handler.load_run_dto()
    experiment, run_id = Run._load_scope()
    current_step_run = Run(experiment, run_id, _run_dto=run_dto)
    logs = run(data_file_path, args, automl_settings, current_step_run)
    metadata_file_handler.write_logs_to_disk(logs)
