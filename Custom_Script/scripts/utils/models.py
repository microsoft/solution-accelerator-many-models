# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def get_model_name(model_type, id):
    id_values = '_'.join([v for k,v in sorted(id.items())])
    return f'{model_type}_{id_values}'
