# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import joblib
from pathlib import Path
import pandas as pd
from azureml.core.model import Model
from azureml.contrib.services.aml_response import AMLResponse


## Input and output sample formats

input_sample = {
    "store": "Store1000", "brand": "dominicks", "model_type": "lr",
    "forecast_horizon": 5, "date_freq": "W-THU",
    "data": {
        "dates": ["2020-04-30", "2020-05-07", "2020-05-14"],
        "values": [11450, 12235, 14713]
    }
}

output_sample = {
    "store": "Store1000", "brand": "dominicks", "model_type": "lr",
    "forecast_horizon": 5, "date_freq": "W-THU",
    "forecast": {
        "dates": ["2020-05-21", "2020-05-28", "2020-06-04", "2020-06-11", "2020-06-18"],
        "values": [14572, 9834, 10512, 12854, 11046]
    }
}


## Webservice definition

def init(): 
    global model_dict
    models_root_path = os.getenv('AZUREML_MODEL_DIR') 
    model_dict = load_all_models(models_root_path)
    print('Models loaded:', model_dict.keys()) 


def run(rawdata):

    metadata, data = format_input_data(json.loads(rawdata))


    # Format prediction dataset
    try:
        prediction_df = format_prediction_data(data, metadata['forecast_horizon'], metadata['date_freq'])
    except ValueError as e:
        return AMLResponse('Wrong input: {}'.format(e), 400)


    # Load model
    try:
        model_name = '{t}_{s}_{b}'.format(t=metadata['model_type'], s=metadata['store'], b=metadata['brand'])
        model = model_dict[model_name]
    except KeyError:
        return AMLResponse('Model not found for store {s} and brand {b} of type {t}'.format(
            s=metadata['store'], b=metadata['brand'], t=metadata['model_type']
        ), 400)
    

    # Forecasting
    for i in range(len(prediction_df)):
        
        x_pred = prediction_df.loc[prediction_df.index == i].drop(columns=['Date', 'Prediction'])
        y_pred = model.predict(x_pred)[0]
        
        prediction_df = update_prediction_data(prediction_df, i, y_pred)
      
    
    result = { 
        **metadata, 
        "forecast": {
            "dates": [d.strftime('%Y-%d-%m') for d in prediction_df.Date],
            "values": prediction_df.Prediction.tolist()
        }
    }

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


def format_input_data(input_data):
    ''' Format data received as input '''
    metadata = {k:v for k,v in input_data.items() if k != 'data'}
    data = pd.DataFrame(input_data['data'])
    data['dates'] = pd.to_datetime(data.dates, format='%Y-%m-%d')
    return metadata, data


def format_prediction_data(data, forecast_horizon, date_freq, nlags=3):
    ''' Format data into the dataset that will be used for prediction '''
        
    dates_past = pd.date_range(end=data.dates.max(), periods=nlags, freq=date_freq)
    if dates_past.isin(data.dates).all():
        data = data.set_index('dates')
        data = data.loc[dates_past]
    else:
        raise ValueError('Expected dates {}'.format(dates_past.strftime("%Y-%m-%d").tolist()))
    
    dates_forecast = pd.date_range(dates_past.max(), periods=forecast_horizon+1, freq=date_freq)[1:]
        
    prediction_df = pd.DataFrame()
    prediction_df['Date'] = dates_forecast
    prediction_df['Prediction'] = None
    prediction_df['Week_Day'] = prediction_df.Date.apply(lambda x: x.weekday())
    for i in range(1, nlags+1):
        prediction_df.loc[0:nlags-1, 'lag_{}'.format(i)] = data.shift(i-nlags).values
    
    return prediction_df


def update_prediction_data(prediction_df, prediction_index, prediction_value, nlags=3):
    ''' Update the dataset used for prediction with the new predictions generated '''
    
    if prediction_index >= len(prediction_df):
        raise ValueError('prediction_index')
    
    prediction_df.loc[prediction_index, 'Prediction'] = prediction_value
    
    total_rows_to_update = min(nlags, len(prediction_df) - prediction_index - 1)
    for i in range(1, total_rows_to_update+1):
        prediction_df.loc[prediction_index+i, 'lag_{}'.format(i)] = prediction_value
    
    return prediction_df
