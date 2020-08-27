# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import glob
import pandas as pd


def upload_to_datastore(datastore, source_path, target_path):
    datastore.upload(src_dir=source_path,
                     target_path=target_path,
                     overwrite=True,
                     show_progress=False)


def split_data_upload_to_datastore(data_path, time_column_name, split_date, datastore, train_ds_target_path,
                                   inference_ds_target_path):

    train_data_path = os.path.join(data_path, "upload_train_data")
    inference_data_path = os.path.join(data_path, "upload_inference_data")
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(inference_data_path, exist_ok=True)

    files_list = [os.path.join(path, f) for path, _, files in os.walk(data_path) for f in files
                  if not path in (train_data_path, inference_data_path)]

    for file in files_list:
        file_name = os.path.basename(file)
        file_extension = os.path.splitext(file_name)[1].lower()
        df = read_file(file, file_extension)

        before_split_date = df[time_column_name] < split_date
        train_df, inference_df = df[before_split_date], df[~before_split_date]

        write_file(train_df, os.path.join(train_data_path, file_name), file_extension)
        write_file(inference_df, os.path.join(inference_data_path, file_name), file_extension)

    upload_to_datastore(datastore, train_data_path, train_ds_target_path)
    upload_to_datastore(datastore, inference_data_path, inference_ds_target_path)


def read_file(path, extension):
    if extension == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def write_file(data, path, extension):
    if extension == ".parquet":
        data.to_parquet(path)
    else:
        data.to_csv(path, index=None, header=True)
