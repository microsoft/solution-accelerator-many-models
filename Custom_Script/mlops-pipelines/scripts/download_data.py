# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
from azureml.opendatasets import OjSalesSimulated


def main(target_path, maxfiles=None):

    # Pull all of the data
    oj_sales_files = OjSalesSimulated.get_file_dataset()

    if maxfiles:
        # Pull only the first <maxfiles> files
        oj_sales_files = oj_sales_files.take(maxfiles)

    # Create a folder to download
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # Download the data
    oj_sales_files.download(target_path, overwrite=True)

    return target_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--maxfiles', type=int)
    args_parsed = parser.parse_args(args)

    if args_parsed.path is None:
        args_parsed.path = 'oj_sales_data{suffix}'.format(
            suffix = '_{}'.format(args_parsed.maxfiles) if args_parsed.maxfiles else ''
        )

    args_parsed.maxfiles = None if args_parsed.maxfiles <= 0 else args_parsed.maxfiles

    return args_parsed


if __name__ == "__main__":
    args = parse_args()
    datapath = main(target_path=args.path, maxfiles=args.maxfiles)
    print(datapath)
