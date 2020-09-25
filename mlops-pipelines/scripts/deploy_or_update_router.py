# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
import warnings
import json
import yaml

from azureml.core import Workspace, Model

from utils.deployment import build_deployment_config, launch_deployment


def main(ws, scripts_dir, config_file, model_name, service_name, aks_target=None):

    deployment_config = build_deployment_config(
        ws, 
        script_dir=scripts_dir,
        script_file='routing_webservice.py',
        environment_file=os.path.join(scripts_dir, 'routing_webservice.conda.yml'),
        config_file=config_file,
        aks_target=aks_target
    )

    routing_model = Model(ws, model_name)

    service = launch_deployment(
        ws, 
        service_name=service_name, 
        models=[routing_model], 
        deployment_config=deployment_config
    )
    print(f'Waiting for deployment of {service.name} to finish...')
    service.wait_for_deployment(show_output=True)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--scripts-dir', required=True, type=str)
    parser.add_argument('--deploy-config-file', required=True, type=str)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--service-name', type=str, default=None)
    parser.add_argument('--aks-target', type=str, default=None)
    args_parsed = parser.parse_args(args)

    if args_parsed.model_name is None:
        args_parsed.model_name = 'test-deployed_models_info' if not args_parsed.aks_target else 'deployed_models_info'

    if args_parsed.service_name is None:
        args_parsed.service_name = 'test-routing-manymodels' if not args_parsed.aks_target else 'routing-manymodels'

    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    # Connect to workspace
    ws = Workspace.get(
        name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group
    )

    main(
        ws,
        scripts_dir=args.scripts_dir,
        config_file=args.deploy_config_file,
        model_name=args.model_name,
        service_name=args.service_name,
        aks_target=args.aks_target
    )
