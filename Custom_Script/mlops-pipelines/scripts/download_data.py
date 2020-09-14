# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import tempfile
import shutil

import pandas as pd
from azureml.opendatasets import OjSalesSimulated


def main(train_path, inference_path, maxfiles=None):

    # Pull all of the data
    oj_sales_files = OjSalesSimulated.get_file_dataset()

    if maxfiles:
        # Pull only the first <maxfiles> files
        oj_sales_files = oj_sales_files.take(maxfiles)

    # Create folders to download
    download_path = tempfile.mkdtemp()
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(inference_path, exist_ok=True)

    # Download the data
    file_paths = oj_sales_files.download(download_path, overwrite=True)

    for fpath in file_paths:
        dataset = pd.read_csv(fpath, dtype={'Store': 'str'})

        # Add extra group columns to data
        dataset['StoreGroup10'] = dataset.Store.str[:-1] + 'X'
        dataset['StoreGroup100'] = dataset.Store.str[:-2] + 'XX'
        dataset['StoreGroup1000'] = dataset.Store.str[:-3] + 'XXX'

        # Split dataset in train/inference subsets
        train_df, inference_df = split_dataset(dataset, 'WeekStarting', '1992-05-28')

        # Write datasets
        train_df.to_csv(os.path.join(train_path, os.path.basename(fpath)), index=False)
        inference_df.to_csv(os.path.join(inference_path, os.path.basename(fpath)), index=False)

    # Clean tmp dir
    shutil.rmtree(download_path)


def split_dataset(df, time_column_name, split_date):
    before_split_date = df[time_column_name] < split_date
    train_df, inference_df = df[before_split_date], df[~before_split_date]
    return train_df, inference_df


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--inference-path', type=str, required=True)
    parser.add_argument('--maxfiles', type=int)
    args_parsed = parser.parse_args(args)

    args_parsed.maxfiles = None if args_parsed.maxfiles <= 0 else args_parsed.maxfiles

    return args_parsed


if __name__ == "__main__":
    args = parse_args()
    main(train_path=args.train_path, inference_path=args.inference_path, maxfiles=args.maxfiles)
    print(f'Train files downloaded into {args.train_path}')
    print(f'Inference files downloaded into {args.inference_path}')
