# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import joblib
from pathlib import Path

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

from utils.webservices import read_input, format_output_record
from utils.models import get_model_name
from utils.forecasting import format_prediction_data


def init():
    global model_dict
    models_root_path = os.getenv('AZUREML_MODEL_DIR')
    model_dict = load_all_models(models_root_path)
    print('Models loaded:', model_dict.keys())


@rawhttp
def run(request):
    if request.method == 'GET':
        return list(model_dict.keys())
    elif request.method == 'POST':
        rawdata = request.get_data(cache=False, as_text=True)
        return serve_forecasting_request(rawdata)
    else:
        return AMLResponse("bad request", 500)


def serve_forecasting_request(rawdata):

    batch = read_input(rawdata, format=True)

    result = []
    for model_record in batch:

        metadata = model_record['metadata']
        data_historical, data_future = model_record['data']
        # print(f'Received request for: {metadata}')

        # Load model
        try:
            model_name = get_model_name(metadata['model_type'], metadata['id'])
            model = model_dict[model_name]
        except KeyError:
            return AMLResponse(f"Model not found  of type {metadata['model_type']} for ID {metadata['id']}", 400)

        # Format prediction dataset
        try:
            prediction_df = format_prediction_data(
                data_historical, data_future, model.time_column_name,
                metadata['forecast_start'], metadata['forecast_freq'], metadata['forecast_horizon']
            )
        except Exception as e:
            return AMLResponse('Wrong input: {}'.format(e), 400)

        # Forecasting
        try:
            forecast = model.forecast(prediction_df)
            forecast = forecast[forecast.index >= metadata['forecast_start']]
        except Exception as e:
            return AMLResponse('Error in model forecasting: {}'.format(e), 400)

        # Append forecasting to output
        model_result = format_output_record(metadata, timestamps=forecast.index, values=forecast.values)
        result.append(model_result)

    return result


def load_all_models(models_root_path):
    ''' Load all the models from the models root dir.
        It returns a dict with key as model name and value as desealized model ready for scoring '''

    model_dict = {}

    models_root_dir = Path(models_root_path)
    models_path_list = [x for x in models_root_dir.iterdir() if x.is_dir()]
    if models_path_list:  # Multiple models deployed
        for model_dir_path in models_path_list:
            model_name = model_dir_path.name
            last_version_dir = max(v for v in model_dir_path.iterdir() if v.is_dir())
            model_dict[model_name] = load_model_via_joblib(last_version_dir, model_name, file_extn='')
    else:  # Single model deployed
        model_name = models_root_dir.parent.name
        model_dict[model_name] = load_model_via_joblib(models_root_dir, model_name, file_extn='')

    return model_dict


def load_model_via_joblib(model_path, model_name, file_extn='pkl'):
    ''' Deserialize and load model '''
    ext = file_extn if file_extn == '' or file_extn.startswith('.') else '.{}'.format(file_extn)
    model_path = model_path.joinpath('{file}{ext}'.format(file=model_name, ext=ext))
    return joblib.load(model_path)
