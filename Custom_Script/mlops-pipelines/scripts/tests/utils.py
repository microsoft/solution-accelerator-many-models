# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class Model:
    def __init__(self, name, version, tags):
        self.name = name
        self.version = version
        self.tags = tags


def create_models_registered(models_info):
    return {info['name']:Model(**info) for info in models_info}


def create_models_deployed(models_info):
    return {info['name']:info for info in models_info}
