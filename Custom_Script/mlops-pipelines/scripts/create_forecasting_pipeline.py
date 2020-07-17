# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from azureml.core import Workspace, Datastore, Dataset, Environment
from azureml.data.data_reference import DataReference
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.contrib.pipeline.steps import ParallelRunConfig, ParallelRunStep


def main(ws, pipeline_name, pipeline_version, dataset_name, output_name, compute_name):

    # Get forecasting dataset
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    dataset_input = dataset.as_named_input(dataset_name)

    # Set outputs
    datastore = ws.get_default_datastore()
    output_dir = PipelineData(name='forecasting_output', datastore=datastore)
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
    parallel_run_config = get_parallel_run_config(ws, dataset_name, compute)
    parallel_run_step = ParallelRunStep(
        name='many-models-parallel-forecasting',
        parallel_run_config=parallel_run_config,
        inputs=[dataset_input],
        output=output_dir,
        allow_reuse=False,
        arguments=[
            '--forecast_horizon', 8,
            '--starting_date', '1992-10-01',
            '--target_column', 'Quantity',
            '--timestamp_column', 'WeekStarting',
            '--model_type', 'lr',
            '--date_freq', 'W-THU'
        ]
    )

    # Create step to copy predictions
    upload_predictions_step = PythonScriptStep(
        name='many-models-copy-predictions',
        source_directory='Custom_Script/scripts/',
        script_name='copy_predictions.py',
        compute_target=compute,
        inputs=[predictions_dref, output_dir],
        allow_reuse=False,
        arguments=[
            '--parallel_run_step_output', output_dir,
            '--output_dir', predictions_dref
        ]
    )

    # Create the pipeline
    train_pipeline = Pipeline(workspace=ws, steps=[parallel_run_step, upload_predictions_step])
    train_pipeline.validate()

    # Publish it and replace old pipeline
    disable_old_pipelines(ws, pipeline_name)
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="Many Models forecasting pipeline",
        version=pipeline_version,
        continue_on_step_failure=False
    )

    return published_pipeline.id


def get_parallel_run_config(ws, dataset_name, compute, processes_per_node=8, node_count=3, timeout=180):

    # Configure environment for ParallelRunStep
    forecast_env = Environment.from_conda_specification(
        name='many_models_environment',
        file_path='Custom_Script/scripts/forecast.conda.yml'
    )

    # Set up ParallelRunStep configuration
    parallel_run_config = ParallelRunConfig(
        source_directory='Custom_Script/scripts/',
        entry_script='forecast.py',
        mini_batch_size='1',
        run_invocation_timeout=timeout,
        error_threshold=25,
        output_action='append_row',
        environment=forecast_env,
        process_count_per_node=processes_per_node,
        compute_target=compute,
        node_count=node_count
    )

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
    parser.add_argument('--name', required=True, type=str)
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
        pipeline_name=args.name,
        pipeline_version=args.version,
        dataset_name=args.dataset,
        output_name=args.output,
        compute_name=args.compute
    )

    print('Forecasting pipeline {} version {} published with ID {}'.format(args.name, args.version, pipeline_id))
