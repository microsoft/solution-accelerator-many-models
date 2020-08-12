# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import argparse
import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compose_logs(file_name, model, start_time):
    logs = []
    logs.append('AutoML')
    logs.append(file_name)
    logs.append(None)
    logs.append(None)
    logs.append(model.name)
    logs.append(model.tags)
    logs.append(start_time)
    logs.append(datetime.datetime.now())
    logs.append(None)
    logs.append(None)
    logs.append(None)
    return logs
