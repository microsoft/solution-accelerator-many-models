# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
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

    files_list = [os.path.join(os.path.abspath(data_path), f) for f in os.listdir(data_path)
                  if os.path.isfile(os.path.join(data_path, f))]

    for file in files_list:
        # print(file)
        file_name = os.path.basename(file)  # file.split('/')[-1][:-4]
        # print(file_name)
        df = pd.read_csv(file)
        df = df.reset_index(drop=True).sort_values(column_name)
        df[column_name] = pd.to_datetime(df[column_name])
        train_df = df[(df[column_name] <= date)]
        test_df = df[(df[column_name] > date)]
        # print(train_df.tail(5))
        # print(test_df.head(5))
        train_df.to_csv(os.path.join(train_data_path, "{}".format(file_name)), index=None, header=True)
        test_df.to_csv(os.path.join(test_data_path, "{}".format(file_name)), index=None, header=True)

    upload_to_datastore(datastore, train_data_path, train_ds_target_path)
    upload_to_datastore(datastore, test_data_path, test_ds_target_path)
