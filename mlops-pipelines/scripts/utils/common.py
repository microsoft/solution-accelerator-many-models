# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import yaml

from azureml.core import Environment


def create_environment_conda(conda_file, environment_variables=None, name='many_models_environment'):

    environment = Environment.from_conda_specification(
        name=name,
        file_path=conda_file
    )
    if environment_variables:
        environment.environment_variables = environment_variables

    return environment


def read_config_file(config_file):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
