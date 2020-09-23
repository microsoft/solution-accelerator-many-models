# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import shutil

from azureml.core import Workspace, Datastore
from azureml.core.compute import AmlCompute
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep

from utils.pipelines import create_parallelrunstep, publish_pipeline


def main(ws, pipeline_name, pipeline_version, dataset_name, output_name, compute_name, 
         scripts_dir, scripts_settings_file, config_file):

    # Get compute target
    compute = AmlCompute(ws, compute_name)

    # Get datastores
    datastore_default = ws.get_default_datastore()
    datastore_predictions = Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=output_name,
        container_name=output_name,
        account_name=datastore_default.account_name,
        account_key=datastore_default.account_key,
        create_if_not_exists=True
    )

    # Setup settings file to be read in script
    settings_filename = os.path.basename(scripts_settings_file)
    shutil.copyfile(scripts_settings_file, os.path.join(scripts_dir, settings_filename))

    # Create the pipeline step for parallel batch forecasting
    parallel_run_step = create_parallelrunstep(
        ws,
        name='many-models-parallel-forecasting',
        compute=compute,
        datastore=datastore_default,
        input_dataset=dataset_name, 
        output_dir='forecasting_output',
        script_dir=scripts_dir,
        script_file='forecast.py',
        environment_file=os.path.join(scripts_dir, 'forecast.conda.yml'), 
        config_file=config_file,
        arguments=[
            '--settings-file', settings_filename
        ]
    )

    # Create the pipeline step to copy predictions
    prev_step_output = parallel_run_step._output[0]
    predictions_dref = DataReference(datastore_predictions)
    upload_predictions_step = PythonScriptStep(
        name='many-models-copy-predictions',
        source_directory=scripts_dir,
        script_name='copy_predictions.py',
        compute_target=compute,
        inputs=[predictions_dref, prev_step_output],
        allow_reuse=False,
        arguments=[
            '--parallel_run_step_output', prev_step_output,
            '--output_dir', predictions_dref,
            '--settings-file', scripts_settings_file
        ]
    )

    # Publish the pipeline
    forecasting_pipeline = publish_pipeline(
        ws, 
        name=pipeline_name, 
        steps=[parallel_run_step, upload_predictions_step],
        description='Many Models forecasting pipeline',
        version=pipeline_version
    )

    return forecasting_pipeline.id


def disable_old_pipelines(ws, pipeline_name):
    for pipeline in PublishedPipeline.list(ws):
        if pipeline.name == pipeline_name:
            pipeline.disable()


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
    parser.add_argument('--dataset', type=str, default='oj_sales_data_inference')
    parser.add_argument('--output', type=str, default='predictions')
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
        output_name=args.output,
        compute_name=args.compute,
        scripts_dir=args.scripts_dir, 
        scripts_settings_file=args.scripts_settings, 
        config_file=args.prs_config
    )

    print('Forecasting pipeline {} version {} published with ID {}'.format(args.name, args.version, pipeline_id))
