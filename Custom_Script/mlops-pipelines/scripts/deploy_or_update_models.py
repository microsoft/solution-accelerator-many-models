# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import warnings
import json
import yaml

from azureml.core import Workspace, Model, Environment, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.exceptions import WebserviceException


def main(ws, config_file, routing_model_name,
         sorting_tags=[], splitting_tags=[], container_size=250,
         aks_target=None, service_prefix='manymodels-', reset=False):

    # Deployment configuration
    deployment_config = get_deployment_config(ws, config_file, aks_target)

    # Get models currently deployed
    models_deployed, existing_services = get_models_deployed(ws, routing_model_name)

    # Reset: delete all webservices and start from scratch
    if reset:
        models_deployed = {}
        for service in existing_services.values():
            service.delete()

    # Get models registered
    models_registered = get_models_registered(ws, exclude_names=[routing_model_name])

    # Get groups to deploy or update and old services to be deleted
    groups_new, groups_update, groups_unchanged, groups_delete = create_deployment_groups(
        models_registered, models_deployed,
        sorting_tags=sorting_tags, splitting_tags=splitting_tags,
        container_size=container_size
    )

    print(f'{len(groups_delete)} services to be deleted,',
          f'{len(groups_new)} new groups to be deployed,',
          f'{len(groups_update)} groups to be updated,',
          f'{len(groups_unchanged)} groups remain unchanged.')

    # Delete old services
    for group_name in groups_delete:
        existing_services[group_name].delete()

    deployments = []

    # Launch webservice deployments
    for group_name, group_models in groups_new.items():
        service = deploy_model_group(ws, group_name, group_models, deployment_config, name_prefix=service_prefix)
        deployments.append({ 'service': service, 'group': group_name, 'models': group_models })

    # Launch webservice updates
    for group_name, group_models in groups_update.items():
        service = deploy_model_group(ws, group_name, group_models, deployment_config, existing_services, name_prefix=service_prefix)
        deployments.append({ 'service': service, 'group': group_name, 'models': group_models })

    models_deployed_updated = { m.name:models_deployed[m.name] for m in groups_unchanged.values() }

    # Wait for deployments to finish
    for deployment in deployments:

        service = deployment['service']
        try:
            print(f'Waiting for deployment of {service.name} to finish...')
            service.wait_for_deployment(show_output=True)
        except WebserviceException as e:
            warnings.warn(f'DEPLOYMENT FAILED FOR SERVICE {service.name}:\n{e}', RuntimeWarning)

        service_info = {
            'webservice': service.name,
            'state': service.state,
            'endpoint': service.scoring_uri if service.state == 'Healthy' else None,
            'key': service.get_keys()[0] if service.auth_enabled and service.state == 'Healthy' else None
        }

        # Store/update deployment info for each deployed model
        for m in deployment['models']:
            models_deployed_updated[m.name] = {
                'version': m.version,
                'group': deployment['group'],
                **service_info
            }

    return models_deployed_updated


def get_models_deployed(ws, routing_model_name):

    # Load deployed models info
    routing_model = Model.list(ws, name=routing_model_name, latest=True)
    if routing_model:
        artifact_path = routing_model[0].download()
        with open(artifact_path, 'r') as f:
            deployed_models = json.load(f)
    else:
        deployed_models = {}

    # Make sure webservices are still deployed
    services = {}
    for group_name, service_name in set((v['group'], v['webservice']) for v in deployed_models.values()):
        try:
            service = Webservice(ws, service_name)
            services[group_name] = service
        except WebserviceException:
            print(f'Webservice {service_name} not found.')

    # Exclude models no longer deployed (associated to deleted webservices)
    deployed_models = { model_name:model_info for model_name, model_info in deployed_models.items()
                        if model_info['webservice'] in services }

    return deployed_models, services


def get_models_registered(ws, exclude_names=[], exclude_tags=[], include_tags=[], page_count=100):

    # Get all models registered in the workspace with specified tags
    all_models = Model.list(ws, tags=include_tags, latest=True, expand=False, page_count=page_count)

    # Exclude models with names or kvtags specified
    models_todeploy = { m.name:m for m in all_models if
                        m.name not in exclude_names and
                        not any(m.tags.get(t) == v for t,v in exclude_tags) }

    print(f'Found {len(models_todeploy)} models registered.')

    return models_todeploy


def create_deployment_groups(models_registered, models_deployed,
                             sorting_tags=[], splitting_tags=[],
                             container_size=250):

    groups_registered = group_models(models_registered.values(), splitting_tags=splitting_tags)
    groups_deployed = get_groups_deployed(models_deployed)

    all_subgroups_registered = {}
    names_all_subgroups_new = []
    names_all_subgroups_update = []
    names_all_subgroups_delete = []

    # Distribute models in groups and subgroups of container_size
    for group_name, group_models_registered in groups_registered.items():

        subgroups_deployed = groups_deployed.get(group_name, {})

        subgroup_models_deployed = [m['name'] for sg in subgroups_deployed.values() for m in sg]
        models_new = [models_registered[mname] for mname in group_models_registered if mname not in subgroup_models_deployed]

        # Update models already in subgroups according to registered models
        subgroups_registered, ind_subgroups_updated, models_relocate = update_existing_models(
            groups=subgroups_deployed,
            models_registered={mname:models_registered[mname] for mname in group_models_registered},
            container_size=container_size
        )

        # Distribute new/relocated models

        models_todeploy = models_new + models_relocate

        if sorting_tags:
            models_todeploy = sorted(models_todeploy, key=lambda m: combine_tags(m, sorting_tags))

        subgroups_registered, ind_subgroups_extended, ind_subgroups_new = distribute_new_models(
            models=models_todeploy,
            existing_groups=subgroups_registered,
            container_size=container_size
        )

        # Add subgroups to global list

        for subgroup_index, subgroup_models in subgroups_registered.items():
            subgroup_name = get_subgroup_name(group_name, subgroup_index)
            all_subgroups_registered[subgroup_name] = subgroup_models

        names_all_subgroups_new += [get_subgroup_name(group_name, i) for i in ind_subgroups_new]
        names_all_subgroups_update += [get_subgroup_name(group_name, i) for i in ind_subgroups_updated | ind_subgroups_extended]
        names_all_subgroups_delete += [get_subgroup_name(group_name, i) for i in subgroups_deployed if i not in subgroups_registered]

    print(f'Grouped models in {len(all_subgroups_registered)} groups.')

    # Split subgroups according to status
    subgroups_new, subgroups_update, subgroups_unchanged = {}, {}, {}
    for name,models in all_subgroups_registered.items():
        subgroups_set = subgroups_new if name in names_all_subgroups_new else \
                        subgroups_update if name in names_all_subgroups_update else \
                        subgroups_unchanged
        subgroups_set[name] = models

    # Find groups with no models registered anymore and mark subgroups for deletion as well
    groups_delete = { gname:sg for gname,sg in groups_deployed.items() if gname not in groups_registered }
    subgroups_delete = names_all_subgroups_delete + [get_subgroup_name(g, i) for g,sg in groups_delete.items() for i in sg]

    return subgroups_new, subgroups_update, subgroups_unchanged, subgroups_delete


def group_models(models, splitting_tags=[]):

    # Create groups of ordered models using splitting tags
    groups = {}
    for m in models:

        if any(t not in m.tags.keys() for t in splitting_tags):
            print(f'Model "{m.name}" does not contain splitting tags. Skipping.')
            continue

        group_name = get_group_name(m, splitting_tags=splitting_tags)
        group = groups.setdefault(group_name, [])
        group.append(m.name)

    return groups


def get_group_name(model, splitting_tags=[]):
    return 'modelgroup' if not splitting_tags else combine_tags(model, splitting_tags)


def combine_tags(model, tags):
    return '-'.join([model.tags.get(t, '') for t in tags])


def get_subgroup_name(group_name, subgroup_index):
    return f'{group_name}-{subgroup_index}'


def get_groups_deployed(models_deployed):

    groups = {}
    for mname, m in models_deployed.items():

        group_name_fields = m['group'].split('-')

        group_name = '-'.join(group_name_fields[:-1])
        group = groups.setdefault(group_name, {})

        subgroup_index = int(group_name_fields[-1])
        subgroup = group.setdefault(subgroup_index, [])
        subgroup.append({'name': mname, 'version': m['version']})

    return groups


def update_existing_models(groups, models_registered, container_size=250):

    groups_updated = {}
    indexes_groups_changed = set()

    models_relocate = []

    # Update models already in groups according to status in registered models
    for group_index, group_models in groups.items():

        group_aml_models = [models_registered.get(m['name']) for m in group_models if m['name'] in models_registered]
        if not group_aml_models:
            continue  # All models have been deleted: delete group as well

        any_deleted = len(group_aml_models) < len(group_models)
        any_changed = any(m.version != models_registered[m.name].version for m in group_aml_models)

        # Put aside models to be deployed elsewhere if group size exceeds maximum
        any_relocate = container_size < len(group_aml_models)
        if any_relocate:
            print(f'Container {group_index} exceeds max container size ({container_size}),',
                  f'{len(group_aml_models) - container_size} models will be relocated.')
            models_relocate += group_aml_models[container_size:]
            group_aml_models = group_aml_models[:container_size]

        if any_deleted or any_changed or any_relocate:
            indexes_groups_changed.add(group_index)

        groups_updated[group_index] = group_aml_models

    return groups_updated, indexes_groups_changed, models_relocate


def distribute_new_models(models, existing_groups, container_size=250):

    #TODO: case when container_size increases

    groups_updated = existing_groups.copy()
    indexes_groups_changed = set()

    indexes_available = sorted(groups_updated.keys())
    try:
        indexes_missing = [i for i in range(1, max(indexes_available)) if i not in indexes_available]
    except ValueError:
        indexes_missing = []

    def next_group():
        if indexes_available:
            index = indexes_available.pop(0)
        elif indexes_missing:
            index = indexes_missing.pop(0)
        else:
            try:
                index = max(groups_updated.keys()) + 1
            except ValueError:
                index = 1
        return index, groups_updated.setdefault(index, [])

    current_index, current_group = next_group()

    # Place models in existing or new groups maintaining max container size
    for m in models:
        while len(current_group) == container_size:
            current_index, current_group = next_group()
        current_group.append(m)
        indexes_groups_changed.add(current_index)

    indexes_groups_new = indexes_groups_changed - existing_groups.keys()
    indexes_groups_extended = indexes_groups_changed - indexes_groups_new

    return groups_updated, indexes_groups_extended, indexes_groups_new


def get_deployment_config(ws, config_file, aks_target=None):

    DEPLOYMENT_TYPES = ['aci', 'aks']

    # Deploy configuration
    deployment_type, deployment_config = get_webservice_config(config_file)

    if deployment_type not in DEPLOYMENT_TYPES:
        raise ValueError('Wrong deployment type. Expected: {}'.format(', '.join(DEPLOYMENT_TYPES)))
    elif deployment_type == 'aks' and aks_target is None:
        raise ValueError('AKS target name needs to be set in AKS deployments')

    deployment_target = AksCompute(ws, aks_target) if deployment_type == 'aks' else None

    # Inference environment
    forecast_env = Environment.from_conda_specification(
        name='many_models_environment',
        file_path='Custom_Script/scripts/forecast_webservice.conda.yml'
    )

    # Inference configuration
    inference_config = InferenceConfig(
        source_directory='Custom_Script/scripts/',
        entry_script='forecast_webservice.py',
        environment=forecast_env
    )

    config = {
        'inference_config': inference_config,
        'deployment_config': deployment_config,
        'deployment_target': deployment_target
    }

    return config


def get_webservice_config(config_file):

    COMPUTE_TYPES = ['aci', 'aks']

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    webservice_type = config['computeType'].lower()
    if webservice_type not in COMPUTE_TYPES:
        raise ValueError('Wrong compute type. Expected: {}'.format(', '.join(COMPUTE_TYPES)))

    compute = AciWebservice if webservice_type == 'aci' else AksWebservice if webservice_type == 'aks' else None
    webservice_config = compute.deploy_configuration(**config['containerResourceRequirements'])

    return webservice_type, webservice_config


def deploy_model_group(ws, group_name, group_models, deployment_config, existing_services={}, name_prefix='manymodels-'):

    if group_name in existing_services:
        service = existing_services[group_name]
        print(f'Launching updating of service {service.name} corresponding to group {group_name}...')
        service.update(
            models=group_models,
            inference_config=deployment_config['inference_config']
        )
        print(f'Updating of {service.name} started')
    else:
        service_name = '{prefix}{group}'.format(
            prefix=name_prefix,
            group=group_name
        ).lower()
        print(f'Launching deployment of service {service_name} corresponding to group {group_name}...')
        service = Model.deploy(
            workspace=ws,
            name=service_name,
            models=group_models,
            **deployment_config,
            overwrite=True
        )
        print(f'Deployment of {service.name} started')

    return service


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', required=True, type=str)
    parser.add_argument('--resource-group', required=True, type=str)
    parser.add_argument('--workspace-name', required=True, type=str)
    parser.add_argument('--deploy-config-file', required=True, type=str)
    parser.add_argument('--splitting-tags', default='', type=lambda str: [t for t in str.split(',') if t])
    parser.add_argument('--sorting-tags', default='', type=lambda str: [t for t in str.split(',') if t])
    parser.add_argument('--routing-model-name', type=str, default='deployed_models_info')
    parser.add_argument('--output', type=str, default='models_deployed.json')
    parser.add_argument('--aks-target', type=str)
    parser.add_argument('--service-prefix', type=str)
    parser.add_argument('--container-size', type=int, default=250)
    parser.add_argument('--reset', action='store_true')
    args_parsed = parser.parse_args(args)

    if args_parsed.aks_target == '':
        args_parsed.aks_target = None

    if args_parsed.service_prefix is None:
        args_parsed.service_prefix = 'test-manymodels-' if not args_parsed.aks_target else 'manymodels-'

    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    # Connect to workspace
    ws = Workspace.get(
        name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group
    )

    models_deployed = main(
        ws,
        config_file=args.deploy_config_file,
        routing_model_name=args.routing_model_name,
        splitting_tags=args.splitting_tags,
        sorting_tags=args.sorting_tags,
        aks_target=args.aks_target,
        service_prefix=args.service_prefix,
        container_size=args.container_size,
        reset=args.reset
    )

    with open(args.output, 'w') as f:
        json.dump(models_deployed, f, indent=4)
