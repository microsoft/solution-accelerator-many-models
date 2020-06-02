# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd


def format_prediction_data(data, forecast_horizon, date_freq, timestamp_column='dates', value_column='values', nlags=3):
    ''' Format data into the dataset that will be used for prediction '''
    
    # Make sure dataset contains all the timestamps needed and is sorted
    timestamps_past = pd.date_range(end=data[timestamp_column].max(), periods=nlags, freq=date_freq)
    if timestamps_past.isin(data[timestamp_column]).all():
        data = data[[timestamp_column, value_column]]
        data = data.set_index(timestamp_column)
        data = data.loc[timestamps_past]
    else:
        raise ValueError('Expected timestamps: {}'.format(timestamps_past.tolist()))
    
    # Calculate forecasting timestamps
    timestamps_forecast = pd.date_range(timestamps_past.max(), periods=forecast_horizon+1, freq=date_freq)[1:]
    
    # Create prediction dataset
    prediction_df = pd.DataFrame()
    prediction_df['Date'] = timestamps_forecast
    prediction_df['Prediction'] = None
    prediction_df['Week_Day'] = prediction_df.Date.apply(lambda x: x.weekday())
    
    # Fill prediction dataset with known lag values
    nrows_tofill = min(forecast_horizon, nlags)
    for i in range(nlags):
        values_lagged = data.shift(-nlags+i+1).values
        prediction_df.loc[:nrows_tofill-1, 'lag_{}'.format(i+1)] = values_lagged[:nrows_tofill]
    
    return prediction_df


def update_prediction_data(prediction_df, prediction_index, prediction_value, nlags=3):
    ''' Update the dataset used for prediction with the new predictions generated '''
    
    if prediction_index >= len(prediction_df):
        raise ValueError('prediction_index')
    
    # Fill prediction cell
    prediction_df.loc[prediction_index, 'Prediction'] = prediction_value
    
    # Fill corresponding lags with prediction value
    index_firstupdate = prediction_index + 1
    nrows_toupdate = min(nlags, len(prediction_df) - index_firstupdate)
    for i in range(nrows_toupdate):
        prediction_df.loc[index_firstupdate+i, 'lag_{}'.format(i+1)] = prediction_value
    
    return prediction_df
