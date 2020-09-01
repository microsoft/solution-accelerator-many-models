import argparse
import os
import json
import shutil
import sys


def validate_parallel_run_config(parallel_run_config):
    max_concurrency = 20
    if (parallel_run_config.process_count_per_node * parallel_run_config.node_count) > max_concurrency:
        print("Please decrease concurrency to maximum of 20 as currently AutoML does not support it.")
        raise ValueError("node_count*process_count_per_node must be between 1 and max_concurrency {}"
                         .format(max_concurrency))


def get_automl_environment():
    from azureml.core import Environment
    from azureml.core.conda_dependencies import CondaDependencies
    from azureml.core.runconfig import DEFAULT_CPU_IMAGE

    train_env = Environment(name="many_models_environment_automl")
    train_conda_deps = CondaDependencies.create(pip_packages=['azureml-sdk[automl]', 'pyarrow==0.14'])
    train_conda_deps.add_pip_package('py-cpuinfo==5.0.0')
    train_conda_deps.add_conda_package('psutil')
    train_conda_deps.add_conda_package('pandas==0.23.4')
    train_conda_deps.add_conda_package('numpy==1.16.2')
    train_conda_deps.add_conda_package('fbprophet==0.5')
    train_conda_deps.add_conda_package('py-xgboost==0.90')
    train_env.python.conda_dependencies = train_conda_deps
    train_env.docker.enabled = True
    train_env.docker.base_image = DEFAULT_CPU_IMAGE
    env = {}
    env['AZUREML_FLUSH_INGEST_WAIT'] = ''
    env['DISABLE_ENV_MISMATCH'] = True
    train_env.environment_variables = env
    return train_env


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
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            keep_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("No files found.")

    # remove empty folders
    if root_path != cur_path:
        os.rmdir(cur_path)
    return
