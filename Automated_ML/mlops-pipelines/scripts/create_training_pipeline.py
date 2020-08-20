# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import sys

from azureml.core import Workspace, Dataset, Environment
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep

sys.path.append("Automated_ML")
sys.path.append("Automated_ML//02b_Train_AutoML")

from common.scripts.helper import get_automl_environment
from scripts.helper import build_parallel_run_config, write_automl_settings_to_file


def main(ws, pipeline_name, pipeline_version, dataset_name, compute_name):

    # Get input dataset
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    dataset_input = dataset.as_named_input(dataset_name)

    # Set output
    datastore = ws.get_default_datastore()
    output_dir = PipelineData(name='training_output', datastore=datastore)

    automl_settings = {
        "task": 'forecasting',
        "primary_metric": 'normalized_root_mean_squared_error',
        "iteration_timeout_minutes": 10,
        "iterations": 15,
        "experiment_timeout_hours": 1,
        "label_column_name": 'Quantity',
        "n_cross_validations": 3,
        "verbosity": logging.INFO,
        "debug_log": 'automl_oj_sales_debug.txt',
        "time_column_name": 'WeekStarting',
        "max_horizon": 6,
        "group_column_names": ['Store', 'Brand'],
        "grain_column_names": ['Store', 'Brand']
    }

    write_automl_settings_to_file(automl_settings)

    # Set up ParallelRunStep
    parallel_run_config = get_parallel_run_config(ws, compute_name)
    parallel_run_step = ParallelRunStep(
        name="many-models-training",
        parallel_run_config=parallel_run_config,
        allow_reuse=False,
        inputs=[dataset_input],
        output=output_dir,
        arguments=[]
        )

    # Create the pipeline
    train_pipeline = Pipeline(workspace=ws, steps=[parallel_run_step])
    train_pipeline.validate()

    # Publish it and replace old pipeline
    disable_old_pipelines(ws, pipeline_name)
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="AutoML Many Models training/retraining pipeline",
        version=pipeline_version,
        continue_on_step_failure=False
    )

    return published_pipeline.id


def get_parallel_run_config(ws, compute_name, node_count=3, processes_count_per_node=8, timeout=300):
    train_env = get_automl_environment()

    # Get the compute target
    compute = AmlCompute(ws, compute_name)

    # build parallel run config
    parallel_run_config = build_parallel_run_config(train_env, compute_name, node_count, processes_count_per_node,
                                                    timeout)
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
    parser.add_argument('--dataset', type=str, default='oj_data_small')
    parser.add_argument('--compute', type=str, default='train-many-model')
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
        compute_name=args.compute
    )

    print('Training pipeline {} version {} published with ID {}'.format(args.pipeline_name, args.version, pipeline_id))
