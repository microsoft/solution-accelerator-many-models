# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import joblib
from pathlib import Path
import pandas as pd
from azureml.core.model import Model
from azureml.contrib.services.aml_response import AMLResponse


from utils.webservices import read_input, format_output_record
from utils.models import get_model_name
from utils.forecasting import format_prediction_data, update_prediction_data


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
            model_name = get_model_name(metadata['store'], metadata['brand'], metadata['model_type'])
            model = model_dict[model_name]
        except KeyError:
            return AMLResponse('Model not found for store {s} and brand {b} of type {t}'.format(
                s=metadata['store'], b=metadata['brand'], t=metadata['model_type']
            ), 400)

        # Format prediction dataset
        try:
            prediction_df = format_prediction_data(data, metadata['forecast_horizon'], metadata['date_freq'])
        except ValueError as e:
            return AMLResponse('Wrong input: {}'.format(e), 400)
    
        # Forecasting
        for i in range(len(prediction_df)):
            x_pred = prediction_df.loc[prediction_df.index == i].drop(columns=['Date', 'Prediction'])
            y_pred = model.predict(x_pred)[0]
            prediction_df = update_prediction_data(prediction_df, i, y_pred)
      
        model_result = format_output_record(metadata, dates=prediction_df.Date, values=prediction_df.Prediction)
        result.append(model_result)

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
        model_dict[model_name] = load_model_via_joblib(last_version_dir, model_name, file_extn='')
    
    return model_dict 


def load_model_via_joblib(model_path, model_name, file_extn='pkl'): 
    ''' Deserialize and load model '''
    ext = file_extn if file_extn == '' or file_extn.startswith('.') else '.{}'.format(file_extn) 
    model_path = model_path.joinpath('{file}{ext}'.format(file=model_name, ext=ext))
    return joblib.load(model_path)
