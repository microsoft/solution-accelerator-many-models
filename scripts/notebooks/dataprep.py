# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import glob
import pandas as pd


def split_data_train_test(data_path, time_column_name, ntest_periods):

    train_data_path = os.path.join(data_path, "upload_train_data")
    test_data_path = os.path.join(data_path, "upload_test_data")

    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)

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
            train_df.to_parquet(os.path.join(train_data_path, file_name))
            test_df.to_parquet(os.path.join(test_data_path, file_name))
        else:
            train_df.to_csv(os.path.join(train_data_path, file_name), index=None, header=True)
            test_df.to_csv(os.path.join(test_data_path, file_name), index=None, header=True)

    return train_data_path, test_data_path
