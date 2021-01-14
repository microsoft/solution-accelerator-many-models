# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import joblib
from pathlib import Path
from azureml.contrib.services.aml_response import AMLResponse

from utils.forecast_helper import get_model_name
from utils.webservices import read_input


def init():
    global model_dict
    models_root_path = os.getenv('AZUREML_MODEL_DIR')
    model_dict = load_all_models(models_root_path)
    print('Models loaded:', model_dict.keys())


def run(rawdata):

    batch = read_input(rawdata, format=True)

    result = []
    for model_record in batch:
        metadata = model_record['metadata']
        data = model_record['data']

        # Load model
        try:
            model_name = get_model_name(metadata, data)
            model = model_dict[model_name]
        except KeyError:
            return AMLResponse('Model not found for'.format(model_name), 400)

        try:
            prediction_df = data.copy()
        except ValueError as e:
            return AMLResponse('Wrong input: {}'.format(e), 400)

        # Forecasting
        y_predictions, X_trans = model.forecast(prediction_df, ignore_data_errors=True)
        predicted_column_name = 'Predictions'
        data[predicted_column_name] = y_predictions

        data_json = data[:].to_json(orient='records', date_format='iso')
        result.append(data_json)

    return result


def load_all_models(models_root_path):
    ''' Load all the models from the models root dir.
        It returns a dict with key as model name and value as desealized model ready for scoring '''

    p = Path(models_root_path)
    models_path_list = [x for x in p.iterdir() if x.is_dir()]
    model_dict = {}

    for model_dir_path in models_path_list:
        model_name = model_dir_path.name

        last_version_dir = max(v for v in model_dir_path.iterdir() if v.is_dir())
        model_dict[model_name] = load_model_via_joblib(last_version_dir, 'model')

    return model_dict


def load_model_via_joblib(model_path, model_name, file_extn='pkl'):
    ''' Deserialize and load model '''
    ext = file_extn if file_extn == '' or file_extn.startswith('.') else '.{}'.format(file_extn)
    model_path = model_path.joinpath('{file}{ext}'.format(file=model_name, ext=ext))
    return joblib.load(model_path)
