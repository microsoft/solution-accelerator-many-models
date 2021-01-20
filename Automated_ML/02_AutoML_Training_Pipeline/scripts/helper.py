# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import sys

from azureml.core import Experiment
from azureml.core.run import Run
from azureml.core import Workspace
sys.path.append("..")


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


def get_automl_environment(workspace: Workspace, automl_settings_dict: dict):
    from common.scripts.helper import get_automl_environment as get_env
    return get_env(workspace, automl_settings_dict)


def get_training_output(run, training_results_name, training_output_name):
    from common.scripts.helper import get_output
    return get_output(run, training_results_name, training_output_name)
