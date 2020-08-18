# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json


def get_automl_environment(name):
    from azureml.core import Environment
    from azureml.core.conda_dependencies import CondaDependencies
    from azureml.core.runconfig import DEFAULT_CPU_IMAGE

    train_env = Environment(name=name)
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
    train_env.environment_variables = env
    return train_env


def validate_parallel_run_config(parallel_run_config):
    errors = False

    if parallel_run_config.mini_batch_size != 1:
        errors = True
        print('Error: mini_batch_size should be set to 1')

    if 'automl' in parallel_run_config.entry_script:
        max_concurrency = 20
        curr_concurrency = parallel_run_config.process_count_per_node * parallel_run_config.node_count
        if curr_concurrency > max_concurrency:
            errors = True
            print(f'Error: node_count*process_count_per_node must be between 1 and max_concurrency {max_concurrency}.',
                  f'Please decrease concurrency from current {curr_concurrency} to maximum of {max_concurrency}',
                  'as currently AutoML does not support it.')

    if not errors:
        print('Validation successful')


def write_automl_settings_to_file(automl_settings):
    with open('scripts//automlconfig.json', 'w', encoding='utf-8') as f:
        json.dump(automl_settings, f, ensure_ascii=False, indent=4)
