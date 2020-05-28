# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd


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
