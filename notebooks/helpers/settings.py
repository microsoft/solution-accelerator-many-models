# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json


def dump_settings(settings, scripts_dir, file_name):
    with open(os.path.join(scripts_dir, file_name), 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
