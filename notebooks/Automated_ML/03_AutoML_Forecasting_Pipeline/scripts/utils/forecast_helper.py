# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import pandas as pd


def get_model_name(metadata, data):
    group_column_names = metadata['group_column_names']

    model_string = '_'.join(str(data[k].iloc[0]) for k in sorted(group_column_names)).lower()
    sha = hashlib.sha256()
    sha.update(model_string.encode())
    model = 'automl_' + sha.hexdigest()
    return model
