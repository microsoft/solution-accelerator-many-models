# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd


def format_prediction_data(data_historical, data_future, timestamp_column,
                           forecast_start, forecast_freq, forecast_horizon):
    ''' Format data into the dataset that will be used for prediction '''

    size_historical = len(data_historical)
    if data_historical.isnull().values.any():
        raise ValueError(f'Historical series must be of the same length, equal to the number of lags used for training')

    size_future = len(data_future)
    if not data_future.empty and size_future != forecast_horizon:
        raise ValueError(f'Future series must be of length {forecast_horizon}, the forecasting horizon')

    ts_historical = pd.date_range(end=forecast_start, periods=size_historical+1, freq=forecast_freq)
    data_historical[timestamp_column] = ts_historical[:-1]
    data_historical.set_index(timestamp_column, inplace=True)

    ts_future = pd.date_range(start=forecast_start, periods=forecast_horizon, freq=forecast_freq)
    data_future[timestamp_column] = ts_future
    data_future.set_index(timestamp_column, inplace=True)

    # Past values of non-lagged data are not needed but can't be empty
    if not data_future.empty:
        for past_timestamp in data_historical.index:
            data_future.loc[past_timestamp] = -1

    full_dataset = data_historical.join(data_future, how='outer')

    return full_dataset
