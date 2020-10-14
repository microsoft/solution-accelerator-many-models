# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import sys
import tempfile
from multiprocessing import current_process
from pathlib import Path
from random import randint
from subprocess import PIPE, Popen
from time import sleep

import pandas as pd
from azureml.automl.core.shared import log_server
from azureml.core import Run

from train_helper import MetadataFileHandler, TrainUtil


# This is used by UI to display the many model settings
many_model_run_properties = {'many_models_run': True}

parser = argparse.ArgumentParser("split")
parser.add_argument("--process_count_per_node", default=1, type=int, help="number of processes per node")
parser.add_argument(
    "--retrain_failed_models", default=False, type=TrainUtil.str2bool, help="retrain failed models only")

args, _ = parser.parse_known_args()


def read_from_json():
    full_path = Path(__file__).absolute().parent
    with open(str(full_path) + "/automlconfig.json") as json_file:
        return json.load(json_file)


automl_settings = read_from_json()
current_step_run = Run.get_context()
metadata_file_handler = MetadataFileHandler(tempfile.mkdtemp())

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
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    working_dir = os.environ.get("AZ_BATCHAI_OUTPUT_logs", "")
    ip_addr = os.environ.get("AZ_BATCHAI_WORKER_IP", "")
    log_dir = os.path.join(working_dir, "user", ip_addr, current_process().name)
    t_log_dir = Path(log_dir)
    t_log_dir.mkdir(parents=True, exist_ok=True)
    automl_settings['many_models'] = True
    automl_settings['many_models_process_count_per_node'] = args.process_count_per_node

    # Try stopping logging server in the parent minibatch process.
    # Otherwise, the logging server will progressively consume more and more CPU, leading to
    # CPU starvation on the box. TODO: diagnose why this happens and fix
    try:
        log_server.server.stop()
    except Exception as e:
        print("Stopping the AutoML logging server in the entry script parent process failed with exception: {}"
              .format(e))

    debug_log = automl_settings.get('debug_log', None)
    if debug_log is not None:
        automl_settings['debug_log'] = os.path.join(log_dir, debug_log)
        automl_settings['path'] = tempfile.mkdtemp()
        print(f"{__file__}.AutoML debug log:{automl_settings['debug_log']}")

    # Write metadata files to disk, so they can be consumed by subprocesses that run AutoML
    metadata_file_handler.write_args_to_disk(args)
    metadata_file_handler.write_automl_settings_to_disk(automl_settings)
    metadata_file_handler.write_run_dto_to_disk(current_step_run._client.run_dto)

    print(f"{__file__}.output_folder:{output_folder}")
    print("init()")
    sleep(randint(1, 120))


def run(input_data_files):
    print("Entering run()")
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    for input_data_file in input_data_files:
        print("Launch subprocess to run AutoML on the data")
        env = os.environ.copy()
        # Aggressively buffer I/O from the subprocess
        env['PYTHONUNBUFFERED'] = '0'
        subprocess = Popen([
            sys.executable,
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_model.py'),
            input_data_file,
            metadata_file_handler.data_dir], env=env, stdout=PIPE, stderr=PIPE)
        for line in subprocess.stdout:
            print(line.decode().rstrip())
        subprocess.wait()
        print("Subprocess completed with exit code: {}".format(subprocess.returncode))
        subprocess_stderr = subprocess.stderr.read().decode().rstrip()
        if subprocess_stderr:
            print("stderr from subprocess:\n{}\n".format(subprocess_stderr))
        if subprocess.returncode != 0:
            raise Exception("AutoML training subprocess exited unsuccesffuly with error code: {}\n"
                            "stderr from subprocess: \n{}\n".format(subprocess.returncode, subprocess_stderr))
        logs = metadata_file_handler.load_logs()
        resultList.append(logs)
        metadata_file_handler.delete_logs_file_if_exists()
    print("Constructing DataFrame from results")
    result = pd.DataFrame(data=resultList)
    print("Ending run()\n")
    return result
