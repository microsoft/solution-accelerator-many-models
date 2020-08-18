# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def get_model_metrics(model):
    return get_run_metrics(model.run, model.name) if model.run else {}


def get_run_metrics(run, model_name=None):

    all_metrics = run.get_metrics()

    # AutoML version
    metrics = {
        'rmse': all_metrics.get('root_mean_squared_error'),
        'mae': all_metrics.get('mean_absolute_error'),
        'mape': all_metrics.get('mean_absolute_percentage_error')
    }

    # Custom script version
    if model_name and not any(metrics.values()):
        metrics = {k:all_metrics.get(f'{model_name}_{k}') for k in metrics}

    return metrics
