# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys

from azureml.core import Workspace, Datastore, Dataset, Environment
from azureml.data.data_reference import DataReference
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep

sys.path.append("Automated_ML")
sys.path.append("Automated_ML//03b_Forecasting_Pipeline")


def main(ws, pipeline_name, pipeline_version, dataset_name, output_name, compute_name):

    # Get forecasting dataset
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    dataset_input = dataset.as_named_input(dataset_name)

    # Set outputs
    datastore = ws.get_default_datastore()
    forecasting_output_name = 'forecasting_output'
    output_dir = PipelineData(name=forecasting_output_name, datastore=datastore)
    predictions_datastore = Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=output_name,
        container_name=output_name,
        account_name=datastore.account_name,
        account_key=datastore.account_key,
        create_if_not_exists=True
    )
    predictions_dref = DataReference(predictions_datastore)

    # Get the compute target
    compute = AmlCompute(ws, compute_name)

    # Set up ParallelRunStep
    parallel_run_config = get_parallel_run_config(ws, compute)
    parallel_run_step = ParallelRunStep(
        name="many-models-forecasting",
        parallel_run_config=parallel_run_config,
        inputs=[dataset_input],
        allow_reuse=False,
        output=output_dir,
        arguments=['--group_column_names', 'Store', 'Brand',
                   '--target_column_name', 'Quantity',
                   '--time_column_name', 'WeekStarting'
                   ]
    )

    # Create the pipeline
    train_pipeline = Pipeline(workspace=ws, steps=parallel_run_step)
    train_pipeline.validate()

    # Publish it and replace old pipeline
    disable_old_pipelines(ws, pipeline_name)
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="AutoML Many Models forecasting pipeline",
        version=pipeline_version,
        continue_on_step_failure=False
    )

    return published_pipeline.id


def get_parallel_run_config(ws, compute_name, node_count=2, processes_count_per_node=8, timeout=300):
    from common.scripts.helper import get_automl_environment
    from scripts.helper import build_parallel_run_config_for_forecasting

    # Configure environment for ParallelRunStep
    forecast_env = get_automl_environment()

    # Set up ParallelRunStep configuration
    parallel_run_config = build_parallel_run_config_for_forecasting(forecast_env, compute_name, node_count,
                                                                    processes_count_per_node, timeout)

    return parallel_run_config


def disable_old_pipelines(ws, pipeline_name):
    for pipeline in PublishedPipeline.list(ws):
        if pipeline.name == pipeline_name:
            pipeline.disable()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--pipeline-name', required=True, type=str)
    parser.add_argument('--version', required=True, type=str)
    parser.add_argument('--dataset', type=str, default='oj_sales_data')
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
        pipeline_name=args.pipeline_name,
        pipeline_version=args.version,
        dataset_name=args.dataset,
        output_name=args.output,
        compute_name=args.compute
    )

    print('Forecasting pipeline {} version {} published with ID {}'.format(args.pipeline_name,
                                                                           args.version, pipeline_id))
