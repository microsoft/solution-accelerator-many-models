# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from collections import defaultdict

import joblib
import requests
from azureml.contrib.services.aml_response import AMLResponse
from requests.exceptions import HTTPError

from utils.forecast_helper import get_model_name
from utils.webservices import read_input


def init():
    global service_mapping
    global service_keys

    models_root_path = os.getenv('AZUREML_MODEL_DIR')
    models_files = [os.path.join(path, f) for path, dirs, files in os.walk(models_root_path) for f in files]
    if len(models_files) > 1:
        raise RuntimeError('Found more than one model')
    routing_model = joblib.load(models_files[0])

    service_mapping = {model: service['endpoint'] for model, service in routing_model.items()}
    service_keys = {service['endpoint']: service['key'] for service in routing_model.values()}


def run(rawdata):

    batch = read_input(rawdata)
    raw_data = read_input(rawdata, format=False)
    # Get forecasting endpoints for all the models in the batch
    services_tocall = defaultdict(list)
    for index, model_data in enumerate(batch):
        # Find model in mapping table
        try:
            metadata = model_data['metadata']
            data = model_data['data']
            model_name = get_model_name(metadata, data)
            model_service = service_mapping[model_name]
        except KeyError:
            return AMLResponse('Model not found for'.format(model_name), 400)

        # Append data to service minibatch
        services_tocall[model_service].append(raw_data[index])

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
