# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
from azureml.core import Workspace, Datastore


def main(ws, train_target_path, test_target_path,
         container_name, account_name, account_key,
         train_data_path_prefix, test_data_path_prefix):

    # Get public datastore
    print(container_name, account_name, account_key)

    if not account_key or account_key == 'None':
        account_key = None

    oj_datastore = get_public_datastore(ws, container_name, account_name, account_key)

    # Create a folder to download
    if not os.path.exists(train_target_path):
        os.mkdir(train_target_path)
    if not os.path.exists(test_target_path):
        os.mkdir(test_target_path)

    # Download the data
    oj_datastore.download(target_path=train_target_path, prefix=train_data_path_prefix, overwrite=True)
    oj_datastore.download(target_path=test_target_path, prefix=test_data_path_prefix, overwrite=True)

    oj_datastore.unregister()

    # Upload the data
    default_datastore = ws.get_default_datastore()

    print("uploading files")
    default_datastore.upload(src_dir=train_target_path,
                             target_path=train_target_path,
                             overwrite=True,
                             show_progress=False)

    default_datastore.upload(src_dir=test_target_path,
                             target_path=test_target_path,
                             overwrite=True,
                             show_progress=False)

    return train_target_path, test_target_path


def get_public_datastore(ws, container_name, account_name, account_key):

    blob_datastore_name = "automl_many_models_ojtestdata"
    oj_datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                           datastore_name=blob_datastore_name,
                                                           container_name=container_name,
                                                           account_name=account_name,
                                                           account_key=account_key,
                                                           create_if_not_exists=True)
    return oj_datastore


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--container-name', required=True, type=str)
    parser.add_argument('--account-name', required=True, type=str)
    parser.add_argument('--account-key', required=False, type=str)
    parser.add_argument('--train-target-path', type=str, default='oj_data_small')
    parser.add_argument('--test-target-path', type=str, default='oj_inference_small')
    parser.add_argument('--train-data-path-prefix', type=str, default='oj_data_small')
    parser.add_argument('--test-data-path-prefix', type=str, default='oj_inference_small')

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

    train_data_path, test_data_path = main(ws, train_target_path=args.train_target_path,
                                           test_target_path=args.test_target_path,
                                           container_name=args.container_name,
                                           account_name=args.account_name,
                                           account_key=args.account_key,
                                           train_data_path_prefix=args.train_data_path_prefix,
                                           test_data_path_prefix=args.test_data_path_prefix)

    print(train_data_path, test_data_path)
