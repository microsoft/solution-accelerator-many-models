import os
import shutil
import sys
from azureml.core import Workspace
from azureml.train.automl._azureautomlsettings import AzureAutoMLSettings


def get_automl_environment(workspace: Workspace, automl_settings_dict: AzureAutoMLSettings):
    from azureml.core import RunConfiguration
    from azureml.train.automl._environment_utilities import modify_run_configuration
    import logging
    null_logger = logging.getLogger("manymodels_null_logger")
    null_logger.addHandler(logging.NullHandler())
    null_logger.propagate = False
    automl_settings_obj = AzureAutoMLSettings.from_string_or_dict(
        automl_settings_dict)
    run_configuration = modify_run_configuration(
        automl_settings_obj,
        RunConfiguration(),
        logger=null_logger)
    train_env = run_configuration.environment
    train_env.environment_variables['DISABLE_ENV_MISMATCH'] = True
    train_env.environment_variables['AZUREML_FLUSH_INGEST_WAIT'] = ''
    train_env.environment_variables['AZUREML_METRICS_POLLING_INTERVAL'] = '30'
    return run_configuration.environment


def get_output(run, results_name, output_name):
    # remove previous run results, if present
    shutil.rmtree(results_name, ignore_errors=True)

    parallel_run_output_file_name = "parallel_run_step.txt"

    # download the contents of the output folder
    batch_run = next(run.get_children())
    batch_output = batch_run.get_output_data(output_name)
    batch_output.download(local_path=results_name)

    keep_root_folder(results_name, results_name)
    for root, dirs, files in os.walk(results_name):
        for file in files:
            if file.endswith(parallel_run_output_file_name):
                result_file = os.path.join(root, file)
                break

    return result_file


def keep_root_folder(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename),
                        os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            keep_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("No files found.")

    # remove empty folders
    if root_path != cur_path:
        os.rmdir(cur_path)
    return
