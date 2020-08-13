# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import sys
import hashlib

sys.path.append("..")


def build_parallel_run_config_for_forecasting(train_env, compute, nodecount, workercount, timeout):
    from azureml.pipeline.steps import ParallelRunConfig
    from common.scripts.helper import validate_parallel_run_config
    parallel_run_config = ParallelRunConfig(
        source_directory='./scripts',
        entry_script='forecast.py',
        mini_batch_size="10",  # do not modify this setting
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


def get_forecasting_output(run, forecasting_results_name, forecasting_output_name):
    from common.scripts.helper import get_output
    return get_output(run, forecasting_results_name, forecasting_output_name)


def get_model_name(tags_dict):
    model_string = '_'.join(str(v) for k, v in sorted(tags_dict.items())).lower()
    sha = hashlib.sha256()
    sha.update(model_string.encode())
    model_name = 'automl_' + sha.hexdigest()
    tags_dict.update({'Hash': sha.hexdigest()})
    return model_name
