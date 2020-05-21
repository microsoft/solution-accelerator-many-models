# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from azureml.core import Workspace, Dataset


def register_dataset(ws, dataset_path, dataset_name):

    # Connect to default datastore
    datastore = ws.get_default_datastore()

    # Create a file dataset
    path_on_datastore = datastore.path(dataset_path)
    ds = Dataset.File.from_files(path=path_on_datastore, validate=False)

    # Register the file dataset
    ds.register(ws, dataset_name, create_new_version=True)

    return dataset_name


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--path', type=str, default='oj_sales_data')
    parser.add_argument('--name', type=str, default='oj_sales_data')
    args_parsed = parser.parse_args(args)
    return args_parsed


if __name__ == "__main__":

    args = parse_args()

    # Connect to workspace
    ws = Workspace.get(
        name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group
    )

    dataset_name = register_dataset(ws, dataset_path=args.path, dataset_name=args.name)
    print(dataset_name)
