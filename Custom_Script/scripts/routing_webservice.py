# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import requests
from requests.exceptions import HTTPError
from collections import defaultdict

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

from utils.webservices import read_input
from utils.models import get_model_name


def init():
    global routing_model_artifact
    global service_mapping
    global service_keys

    models_root_path = os.getenv('AZUREML_MODEL_DIR')
    models_files = [os.path.join(path, f) for path,dirs,files in os.walk(models_root_path) for f in files]
    if len(models_files) > 1:
        raise RuntimeError(f'Found more than one model: {models_files}')
    with open(models_files[0], 'r') as f:
        routing_model_artifact = json.load(f)

    service_mapping = {model:service['endpoint'] for model,service in routing_model_artifact.items()}
    service_keys = {service['endpoint']:service['key'] for service in routing_model_artifact.values()}


@rawhttp
def run(request):
    if request.method == 'GET':
        return routing_model_artifact
    elif request.method == 'POST':
        rawdata = request.get_data(cache=False, as_text=True)
        print(rawdata)
        return route_forecasting_requests(rawdata)
    else:
        return AMLResponse("bad request", 500)


def route_forecasting_requests(rawdata):
    ''' Call the appropiate model webservice for all the records in the request body '''

    batch = read_input(rawdata, format=False)

    # Get forecasting endpoints for all the models in the batch
    services_tocall = defaultdict(list)
    for model_data in batch:
        # Find model in mapping table
        try:
            model_name = get_model_name(model_data['model_type'], model_data['id'])
            model_service = service_mapping[model_name]
        except KeyError:
            return AMLResponse(f"Model not found  of type {model_data['model_type']} for ID {model_data['id']}", 400)
        # Append data to service minibatch
        services_tocall[model_service].append(model_data)

    # Call endpoints and store result
    result = []
    for service, minibatch in services_tocall.items():
        try:
            response = call_model_webservice(service, minibatch, service_key=service_keys[service])
            response.raise_for_status()
        except HTTPError:
            return AMLResponse(response.text, response.status_code)
        result += response.json()

    return result


def call_model_webservice(service_endpoint, data, service_key=None):
    ''' Call the model webservice to get the forecasting '''

    request_headers = {'Content-Type': 'application/json'}
    if service_key:
        request_headers['Authorization'] = f'Bearer {service_key}'

    response = requests.post(service_endpoint, json=data, headers=request_headers)

    return response
