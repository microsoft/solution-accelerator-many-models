import argparse
import json
import sys

sys.path.append("..")


def write_automl_settings_to_file(automl_settings):
    with open('scripts//automlconfig.json', 'w', encoding='utf-8') as f:
        json.dump(automl_settings, f, ensure_ascii=False, indent=4)


def build_parallel_run_config(train_env, compute, nodecount, workercount, timeout):
    from azureml.contrib.pipeline.steps import ParallelRunConfig
    from common.scripts.helper import validate_parallel_run_config
    parallel_run_config = ParallelRunConfig(
        source_directory='./scripts',
        entry_script='train_automl.py',
        mini_batch_size="1",  # do not modify this setting
        run_invocation_timeout=timeout,
        error_threshold=100,
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
