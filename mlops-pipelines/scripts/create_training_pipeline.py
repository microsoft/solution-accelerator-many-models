# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import shutil

from azureml.core import Workspace
from azureml.core.compute import AmlCompute

from utils.pipelines import create_parallelrunstep, publish_pipeline


def main(ws, pipeline_name, pipeline_version, dataset_name, compute_name, 
         scripts_dir, scripts_settings_file, config_file):

    # Get the compute target
    compute = AmlCompute(ws, compute_name)

    # Get datastore
    datastore_default = ws.get_default_datastore()

    # Setup settings file to be read in script
    settings_filename = os.path.basename(scripts_settings_file)
    shutil.copyfile(scripts_settings_file, os.path.join(scripts_dir, settings_filename))

    # Create the pipeline step for parallel training
    parallel_run_step = create_parallelrunstep(
        ws,
        name='many-models-parallel-training',
        compute=compute,
        datastore=datastore_default,
        input_dataset=dataset_name, 
        output_dir='training_output',
        script_dir=scripts_dir,
        script_file='train.py',
        environment_file=os.path.join(scripts_dir, 'train.conda.yml'),
        config_file=config_file,
        arguments=[
            '--settings-file', settings_filename
        ]
    )

    # Publish the pipeline
    train_pipeline = publish_pipeline(
        ws, 
        name=pipeline_name, 
        steps=[parallel_run_step],
        description='Many Models training/retraining pipeline',
        version=pipeline_version
    )

    return train_pipeline.id


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--version', required=True, type=str)
    parser.add_argument('--scripts-dir', required=True, type=str)
    parser.add_argument('--scripts-settings', required=True, type=str)
    parser.add_argument('--prs-config', required=True, type=str)
    parser.add_argument('--dataset', type=str, default='oj_sales_data_train')
    parser.add_argument('--compute', type=str, default='cpu-compute')
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

    pipeline_id = main(
        ws,
        pipeline_name=args.name,
        pipeline_version=args.version,
        dataset_name=args.dataset,
        compute_name=args.compute,
        scripts_dir=args.scripts_dir, 
        scripts_settings_file=args.scripts_settings, 
        config_file=args.prs_config
    )

    print('Training pipeline {} version {} published with ID {}'.format(args.name, args.version, pipeline_id))
