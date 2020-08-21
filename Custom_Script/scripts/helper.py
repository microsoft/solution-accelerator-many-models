# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import glob
import sys
import pandas as pd
sys.path.append("..")


def build_parallel_run_config_for_forecasting(train_env, compute, nodecount, workercount, timeout):
    from azureml.pipeline.steps import ParallelRunConfig
    from common.scripts.helper import validate_parallel_run_config
    parallel_run_config = ParallelRunConfig(
        source_directory='./scripts',
        entry_script='forecast.py',
        mini_batch_size="1",  # do not modify this setting
        run_invocation_timeout=timeout,
        error_threshold=10,
        output_action="append_row",
        environment=train_env,
        process_count_per_node=workercount,
        compute_target=compute,
        node_count=nodecount)
    validate_parallel_run_config(parallel_run_config)
    return parallel_run_config


def get_automl_environment():
    from common.scripts.helper import get_automl_environment as get_env
    return get_env()

def upload_to_datastore(datastore, source_path, target_path):
    datastore.upload(src_dir=source_path,
                     target_path=target_path,
                     overwrite=True,
                     show_progress=False)

def split_data_upload_to_datastore(data_path, time_column_name, ntest_periods, datastore, train_ds_target_path,
                                   test_ds_target_path):
    train_data_path = data_path + "/upload_train_data"
    test_data_path = data_path + "/upload_test_data"
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    files_list = [f for f in set(glob.glob(os.path.join(data_path, "**"), recursive=True)) -
                  set(glob.glob(os.path.join(data_path, "upload_**"), recursive=True))
                  if os.path.isfile(f)]

    for file in files_list:
        file_name = os.path.basename(file)
        file_extension = os.path.splitext(file_name)[1]
        if file_extension.lower() == ".parquet":
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)

        df.reset_index(drop=True, inplace=True)
        df[time_column_name] = pd.to_datetime(df[time_column_name])
        df.sort_values(time_column_name, ascending=True, inplace=True)
        train_df = df.iloc[:-ntest_periods].copy()
        test_df = df.iloc[-ntest_periods:].copy()

        if file_extension.lower() == ".parquet":
            train_df.to_parquet(os.path.join(train_data_path, "{}".format(
                file_name)))
            test_df.to_parquet(os.path.join(test_data_path, "{}".format(
                file_name)))
        else:
            train_df.to_csv(os.path.join(train_data_path, "{}".format(
                file_name)), index=None, header=True)
            test_df.to_csv(os.path.join(test_data_path, "{}".format(
                file_name)), index=None, header=True)

    upload_to_datastore(datastore, train_data_path, train_ds_target_path)
    upload_to_datastore(datastore, test_data_path, test_ds_target_path)
