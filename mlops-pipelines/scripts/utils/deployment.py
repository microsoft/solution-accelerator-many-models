# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import yaml

from azureml.core import Environment, Model, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.exceptions import WebserviceException


def build_deployment_config(ws, script_dir, script_file, environment_file, config_file, aks_target=None):

    COMPUTE_TYPES = ['aci', 'aks']

    compute_type, compute_config, environment_config = read_config(config_file)

    # Deployment target
    if compute_type not in COMPUTE_TYPES:
        raise ValueError(f'Wrong compute type in {config_file}. Expected: {", ".join(COMPUTE_TYPES)}')
    elif compute_type == 'aks' and aks_target is None:
        raise ValueError('AKS target name needs to be set in AKS deployments')
    deployment_target = AksCompute(ws, aks_target) if compute_type == 'aks' else None

    # Inference environment
    environment = Environment.from_conda_specification(
        name='many_models_environment',
        file_path=environment_file
    )
    if environment_config:
        environment.environment_variables = environment_config

    # Inference configuration
    inference_config = InferenceConfig(
        source_directory=script_dir,
        entry_script=script_file,
        environment=environment
    )

    # Set max concurrent requests in AKS
    if compute_type == 'aks':
        worker_count = environment.environment_variables.get('WORKER_COUNT', 1)
        max_requests = compute_config.get('replica_max_concurrent_requests')
        if max_requests and max_requests != worker_count:
            raise ValueError(f'replica_max_concurrent_requests and WORKER_COUNT should have the same value in {config_file}')
        compute_config['replica_max_concurrent_requests'] = worker_count

    # Deploy configuration
    compute = AciWebservice if compute_type == 'aci' else AksWebservice if compute_type == 'aks' else None
    deployment_config = compute.deploy_configuration(**compute_config)

    config = {
        'inference_config': inference_config,
        'deployment_config': deployment_config,
        'deployment_target': deployment_target
    }

    return config


def read_config(config_file):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    compute_type = config['computeType'].lower()
    webservice_config = config['containerResourceRequirements']
    environment_config = config.get('environmentVariables', {})

    return compute_type, webservice_config, environment_config


def launch_deployment(ws, service_name, models, deployment_config, existing_services=None):

    # Try to get service if haven't done before
    if existing_services is None:
        try:
            service = Webservice(ws, service_name)
            existing_services = {service_name: service}
        except WebserviceException:
            existing_services = {}

    # Update or deploy service
    service = existing_services.get(service_name)
    if service:
        print(f'Launching updating of service {service.name}...')
        service.update(
            models=models,
            inference_config=deployment_config['inference_config']  
        )
        print(f'Updating of {service.name} started')
    else:
        print(f'Launching deployment of service {service_name}...')
        service = Model.deploy(
            workspace=ws,
            name=service_name,
            models=models,
            **deployment_config,
            overwrite=True
        )
        print(f'Deployment of {service.name} started')

    return service
