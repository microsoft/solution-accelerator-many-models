# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import joblib
import requests
from requests.exceptions import HTTPError
from collections import defaultdict
from azureml.core.model import Model
from azureml.contrib.services.aml_response import AMLResponse

from utils.webservices import read_input
from utils.models import get_model_name


def init(): 
    global service_mapping
    global service_keys
    
    models_root_path = os.getenv('AZUREML_MODEL_DIR') 
    models_files = [os.path.join(path, f) for path,dirs,files in os.walk(models_root_path) for f in files]
    if len(models_files) > 1:
        raise RuntimeError('Found more than one model')
    routing_model = joblib.load(models_files[0])

    service_mapping = { model:service['endpoint'] for model,service in routing_model.items()}
    service_keys = { service['endpoint']:service['key'] for service in routing_model.values()}


def run(rawdata):

    batch = read_input(rawdata, format=False)


    ## Get forecasting endpoints for all the models in the batch
    services_tocall = defaultdict(list)
    for model_data in batch:
        
        # Find model in mapping table
        try:
            model_name = get_model_name(model_data['store'], model_data['brand'], model_data['model_type'])
            model_service = service_mapping[model_name]
        except KeyError:
            return AMLResponse('Model not found for store {s} and brand {b} of type {t}'.format(
                s=model_data['store'], b=model_data['brand'], t=model_data['model_type']
            ), 400)
        
        # Append data to service minibatch
        services_tocall[model_service].append(model_data)


    ## Call endpoints and store result
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
        request_headers['Authorization'] = 'Bearer f{service_key}'
    
    response = requests.post(service_endpoint, json=data, headers=request_headers)

    return response
