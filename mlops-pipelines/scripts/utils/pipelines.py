# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Environment, Dataset
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
 
from utils.common import read_config_file, create_environment_conda


def create_parallelrunstep(ws, name, compute, datastore, input_dataset, output_dir, 
                           script_dir, script_file, environment_file, config_file, arguments):

    config_params = read_config_file(config_file)
    
    # Get input dataset
    dataset = Dataset.get_by_name(ws, name=input_dataset)
    pipeline_input = dataset.as_named_input(input_dataset)

    # Set output
    pipeline_output = PipelineData(name=output_dir, datastore=datastore)

    # Create ParallelRunConfig
    parallel_run_config = create_parallelrunconfig(
        ws, compute, script_dir, script_file, environment_file, config_params)

    parallel_run_step = ParallelRunStep(
        name=name,
        parallel_run_config=parallel_run_config,
        inputs=[pipeline_input],
        output=pipeline_output,
        allow_reuse=False,
        arguments=arguments
    )

    return parallel_run_step


def create_parallelrunconfig(ws, compute, script_dir, script_file, environment_file, config_params):

    # Configure environment for ParallelRunStep
    train_env = create_environment_conda(environment_file)

    # Set up ParallelRunStep configuration

    default_params = {
        'mini_batch_size': '1',
        'run_invocation_timeout': 300,
        'error_threshold': -1,
        'process_count_per_node': 8,
        'node_count': 3
    }
    for param, default_value in default_params.items():
        if not param in config_params:
            config_params[param] = default_value

    parallel_run_config = ParallelRunConfig(
        source_directory=script_dir,
        entry_script=script_file,
        environment=train_env,
        compute_target=compute,
        output_action='append_row',
        **config_params
    )

    return parallel_run_config


def create_step_pythonscript():
    pass


def publish_pipeline(ws, name, steps, description=None, version=None):

    # Create the pipeline
    pipeline = Pipeline(workspace=ws, steps=steps)
    pipeline.validate()

    # Publish it replacing old pipeline
    disable_old_pipelines(ws, name)
    published_pipeline = pipeline.publish(
        name=name,
        description=description,
        version=version,
        continue_on_step_failure=False
    )

    return published_pipeline


def disable_old_pipelines(ws, name):
    for pipeline in PublishedPipeline.list(ws):
        if pipeline.name == name:
            pipeline.disable()
