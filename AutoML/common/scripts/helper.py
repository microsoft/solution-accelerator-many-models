import argparse
import json


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
    train_conda_deps = CondaDependencies.create(pip_packages=['azureml-sdk[automl]', 'joblib', 'pyarrow==0.14'])
    train_conda_deps.add_conda_package('pandas==0.23.4')
    train_conda_deps.add_conda_package('numpy==1.16.2')
    train_conda_deps.add_conda_package('fbprophet')
    train_conda_deps.add_conda_package('py-xgboost==0.80')
    train_env.python.conda_dependencies = train_conda_deps
    train_env.docker.enabled = True
    train_env.docker.base_image = DEFAULT_CPU_IMAGE
    return train_env
