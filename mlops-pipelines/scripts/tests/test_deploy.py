# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from deploy_or_update_models import create_deployment_groups

from tests.utils import create_models_registered, create_models_deployed


def test_grouping():

    mdeployed = create_models_deployed([
        {'name': 'A11', 'version': 1, 'group': 'A-1'}, {'name': 'A12', 'version': 1, 'group': 'A-1'},
        {'name': 'A13', 'version': 1, 'group': 'A-1'}, {'name': 'A14', 'version': 1, 'group': 'A-1'},
        {'name': 'A21', 'version': 1, 'group': 'A-2'}, {'name': 'A22', 'version': 1, 'group': 'A-2'},
        {'name': 'A23', 'version': 1, 'group': 'A-2'}, {'name': 'A24', 'version': 1, 'group': 'A-2'},
        {'name': 'A31', 'version': 1, 'group': 'A-3'}, {'name': 'A32', 'version': 1, 'group': 'A-3'},
        {'name': 'A33', 'version': 1, 'group': 'A-3'}, {'name': 'A34', 'version': 1, 'group': 'A-3'},
        {'name': 'A41', 'version': 1, 'group': 'A-4'}, {'name': 'A42', 'version': 1, 'group': 'A-4'},
        {'name': 'B11', 'version': 1, 'group': 'B-1'}, {'name': 'B12', 'version': 1, 'group': 'B-1'},
        {'name': 'B13', 'version': 1, 'group': 'B-1'}, {'name': 'B14', 'version': 1, 'group': 'B-1'},
        {'name': 'B21', 'version': 1, 'group': 'B-2'}, {'name': 'B22', 'version': 1, 'group': 'B-2'},
        {'name': 'B23', 'version': 1, 'group': 'B-2'},
        {'name': 'B31', 'version': 1, 'group': 'B-3'}, {'name': 'B32', 'version': 1, 'group': 'B-3'},
        {'name': 'C11', 'version': 1, 'group': 'C-1'}, {'name': 'C12', 'version': 1, 'group': 'C-1'}
    ])

    mregistered = create_models_registered([
        {'name': 'A11', 'version': 2, 'tags': {'split': 'A', 'sort': '1'}}, {'name': 'A12', 'version': 2, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'A14', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'A23', 'version': 2, 'tags': {'split': 'A', 'sort': '1'}}, {'name': 'A24', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'A31', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}}, {'name': 'A32', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'A33', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}}, {'name': 'A34', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'A41', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}}, {'name': 'A42', 'version': 1, 'tags': {'split': 'A', 'sort': '1'}},
        {'name': 'Anew1', 'version': 1, 'tags': {'split': 'A', 'sort': '0'}}, {'name': 'Anew2', 'version': 1, 'tags': {'split': 'A', 'sort': '0'}},
        {'name': 'B12', 'version': 1, 'tags': {'split': 'A', 'sort': '4'}},
        {'name': 'B11', 'version': 2, 'tags': {'split': 'B', 'sort': '1'}},
        {'name': 'B13', 'version': 1, 'tags': {'split': 'B', 'sort': '1'}}, {'name': 'B14', 'version': 1, 'tags': {'split': 'B', 'sort': '1'}},
        {'name': 'B31', 'version': 1, 'tags': {'split': 'B', 'sort': '1'}}, {'name': 'B32', 'version': 1, 'tags': {'split': 'B', 'sort': '1'}},
        {'name': 'Dnew1', 'version': 1, 'tags': {'split': 'D', 'sort': '1'}}, {'name': 'Dnew2', 'version': 1, 'tags': {'split': 'D', 'sort': '1'}}
    ])

    groups_new, groups_update, groups_unchanged, groups_delete = create_deployment_groups(
        models_registered=mregistered,
        models_deployed=mdeployed,
        splitting_tags=['split'], sorting_tags=['sort'],
        container_size=3
    )

    assert list(groups_new) == ['A-5', 'D-1']

    assert get_model_names(groups_new['A-5']) == ['A34', 'B12']
    assert get_model_versions(groups_new['A-5']) == [1, 1]

    assert list(groups_update) == ['A-1', 'A-2', 'A-3', 'A-4', 'B-1']

    assert get_model_names(groups_update['A-1']) == ['A11', 'A12', 'A14']
    assert get_model_versions(groups_update['A-1']) == [2, 2, 1]

    assert get_model_names(groups_update['A-2']) == ['A23', 'A24', 'Anew1']
    assert get_model_versions(groups_update['A-2']) == [2, 1, 1]

    assert get_model_names(groups_update['A-3']) == ['A31', 'A32', 'A33']
    assert get_model_versions(groups_update['A-3']) == [1, 1, 1]

    assert get_model_names(groups_update['A-3']) == ['A31', 'A32', 'A33']
    assert get_model_versions(groups_update['A-3']) == [1, 1, 1]

    assert get_model_names(groups_update['A-4']) == ['A41', 'A42', 'Anew2']
    assert get_model_versions(groups_update['A-4']) == [1, 1, 1]

    assert get_model_names(groups_update['B-1']) == ['B11', 'B13', 'B14']
    assert get_model_versions(groups_update['B-1']) == [2, 1, 1]

    assert list(groups_unchanged) == ['B-3']

    assert get_model_names(groups_unchanged['B-3']) == ['B31', 'B32']
    assert get_model_versions(groups_unchanged['B-3']) == [1, 1]

    assert groups_delete == ['B-2', 'C-1']


def get_model_names(group):
    return [m.name for m in group]


def get_model_versions(group):
    return [m.version for m in group]
