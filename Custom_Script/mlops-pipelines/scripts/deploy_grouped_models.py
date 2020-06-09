# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import argparse
import joblib
from azureml.core import Workspace, Model, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute
from azureml.core.webservice import AciWebservice, AksWebservice


DEPLOYMENT_TYPES = ['aci', 'aks']


def main(ws, deployment_type, grouping_tags=None, exclude=[], aks_target=None):

    if deployment_type not in DEPLOYMENT_TYPES:
        raise ValueError('Wrong deployment type. Expected: {}'.format(', '.join(DEPLOYMENT_TYPES)))

    grouped_models = get_grouped_models(grouping_tags, exclude=exclude)
    deployment_config = get_deployment_config(deployment_type, aks_target)

    # Deploy groups
    endpoints = {}
    for group_name, group_models in grouped_models.items():
        service = deploy_model_group(ws, deployment_type, group_name, group_models, deployment_config)
        
        # Store pairs model - endpoint where the model is deployed
        for m in group_models:
            endpoints[m.name] = { 
                'endpoint': service.scoring_uri, 
                'key': service.get_keys()[0] if service.auth_enabled else None
            }

    return endpoints


def get_grouped_models(grouping_tags=None, exclude=[]):
    
    # Get all models registered in the workspace
    all_models = Model.list(ws, latest=True)

    # Group models by tags
    grouped_models = {}
    for m in all_models:
        # Exclude models that follow certain conditions (meta-models)
        if any(m.tags[t] == v for t,v in exclude):
            continue
        # Create or update group
        group_name = '/'.join([m.tags[t] for t in grouping_tags]) if grouping_tags is not None else m.name
        group = grouped_models.setdefault(group_name, [])
        group.append(m)
    
    return grouped_models


def get_deployment_config(deployment_type, aks_target=None, cores=1, memory=1):
    
    if deployment_type == 'aks' and aks_target is None:
        raise ValueError('AKS target name needs to be set in AKS deployments')

    # Define inference environment
    forecast_env = Environment(name="many_models_environment")
    forecast_conda_deps = CondaDependencies.create(pip_packages=['azureml-defaults', 'sklearn'])
    forecast_env.python.conda_dependencies = forecast_conda_deps
    
    # Define inference configuration
    inference_config = InferenceConfig(
        source_directory='Custom_Script/scripts/',
        entry_script='forecast_webservice.py',
        environment=forecast_env
    )

    # Define deploy configuration
    if deployment_type == 'aci':
        deployment_config = AciWebservice.deploy_configuration(cpu_cores=cores, memory_gb=memory)
        deployment_target = None
    elif deployment_type == 'aks':
        deployment_config = AksWebservice.deploy_configuration(cpu_cores=cores, memory_gb=memory)
        deployment_target = AksCompute(ws, aks_target)
    
    config = {
        'inference_config': inference_config,
        'deployment_config': deployment_config,
        'deployment_target': deployment_target
    }
    
    return config


def deploy_model_group(ws, deployment_type, group_name, group_models, deployment_config):
    
    service_name = '{prefix}manymodels-{group}'.format(
        prefix='test-' if deployment_type == 'aci' else '',
        group=group_name
    ).lower()

    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=group_models,
        **deployment_config,
        overwrite=True
    )
    
    print('Deploying {}...'.format(service_name))
    service.wait_for_deployment(show_output=True)
    assert service.state == 'Healthy'
    
    return service


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument("--grouping-tags", type=lambda str: [t for t in str.split(',') if t])
    parser.add_argument("--routing-model-tag-name", type=str, default='ModelType')
    parser.add_argument("--routing-model-tag-value", type=str, default='_meta_')
    parser.add_argument("--endpoints-path", type=str, default='endpoints.pkl')
    parser.add_argument("--aks-target", type=str)
    args_parsed = parser.parse_args(args)

    if args_parsed.aks_target == '':
        args_parsed.aks_target = None
    
    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    # Connect to workspace
    ws = Workspace.get(
        name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group
    )

    routing_model_tags = [(args.routing_model_tag_name, args.routing_model_tag_value)]

    deployment_type = 'aks' if args.aks_target else 'aci'

    endpoints = main(
        ws, 
        deployment_type=deployment_type,
        grouping_tags=args.grouping_tags, 
        exclude=routing_model_tags,
        aks_target=args.aks_target
    )
    
    joblib.dump(endpoints, args.endpoints_path)
