# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import argparse
import json
import sys

from azureml.core import Experiment
from azureml.core.run import Run

sys.path.append("..")


def write_automl_settings_to_file(automl_settings):
    with open('scripts//automlconfig.json', 'w', encoding='utf-8') as f:
        json.dump(automl_settings, f, ensure_ascii=False, indent=4)


def cancel_runs_in_experiment(ws, experiment):
    failed_experiment = Experiment(ws, experiment)
    all_runs = failed_experiment.get_runs()
    for idx, run in enumerate(all_runs):
        try:
            if run.status == 'Running':
                run = Run(failed_experiment, run.id)
                print('Canceling run: ', run)
                run.cancel()
        except Exception as e:
            print('Canceling run failed due to ', e)


def build_parallel_run_config(train_env, compute, nodecount, workercount, timeout):
    from azureml.pipeline.steps import ParallelRunConfig
    from common.scripts.helper import validate_parallel_run_config
    parallel_run_config = ParallelRunConfig(
        source_directory='./scripts',
        entry_script='train_automl.py',
        mini_batch_size="1",  # do not modify this setting
        run_invocation_timeout=timeout,
        error_threshold=-1,
        output_action="append_row",
        environment=train_env,
        process_count_per_node=workercount,
        compute_target=compute,
        node_count=nodecount)
    validate_parallel_run_config(parallel_run_config)
    return parallel_run_config


def get_automl_environment():
    from common.scripts.helper import get_automl_environment as get_env
    return get_env()


def get_training_output(run, training_results_name, training_output_name):
    from common.scripts.helper import get_output
    return get_output(run, training_results_name, training_output_name)
