import os
import shutil
import sys

sys.path.append("..")


def build_parallel_run_config_for_forecasting(train_env, compute, nodecount, workercount, timeout):
    from azureml.pipeline.steps import ParallelRunConfig
    from common.scripts.helper import validate_parallel_run_config
    parallel_run_config = ParallelRunConfig(
        source_directory='./scripts',
        entry_script='forecast.py',
        mini_batch_size="10",  # do not modify this setting
        run_invocation_timeout=timeout,
        error_threshold=10,
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
    # remove previous run results, if present
    shutil.rmtree(forecasting_results_name, ignore_errors=True)

    parallel_run_output_file_name = "parallel_run_step.txt"

    # download the contents of the output folder
    batch_run = next(run.get_children())
    batch_output = batch_run.get_output_data(forecasting_output_name)
    batch_output.download(local_path=forecasting_results_name)

    keep_root_folder(forecasting_results_name, forecasting_results_name)
    for root, dirs, files in os.walk(forecasting_results_name):
        for file in files:
            if file.endswith(parallel_run_output_file_name):
                result_file = os.path.join(root, file)
                break

    return result_file


def keep_root_folder(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            keep_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("No files found.")

    # remove empty folders
    if root_path != cur_path:
        os.rmdir(cur_path)
    return
