# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import glob
import pandas as pd

from azureml.core import Dataset, Datastore


def upload_to_datastore(datastore, source_path, target_path):
    datastore.upload(src_dir=source_path,
                     target_path=target_path,
                     overwrite=True,
                     show_progress=False)


def split_data_upload_to_datastore(data_path, column_name, date, datastore, train_ds_target_path,
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

        df = df.reset_index(drop=True).sort_values(column_name)
        df[column_name] = pd.to_datetime(df[column_name])
        train_df = df[(df[column_name] <= date)]
        test_df = df[(df[column_name] > date)]

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
