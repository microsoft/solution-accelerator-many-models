# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import argparse
from sklearn.externals import joblib
from azureml.core.model import Model
from azureml.core.run import Run
import datetime

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forecasting") #change this to tak the last date and start from there
parser.add_argument("--target_column", type=str, help="target colunm to predict on")
parser.add_argument("--timestamp_column", type=str, help="timestamp column from data")
parser.add_argument("--model_type", type=str, help="model type")
parser.add_argument("--date_freq", type=str, help="the step size for predictions, daily, weekly")

args, _ = parser.parse_known_args()


def run(input_data):
    # 1.0 Set up results dataframe
    results = pd.DataFrame()

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        # 3.0 Set up data to predict on
        data = pd.read_csv(csv_file_path, header = 0)
        data[args.timestamp_column] = data[args.timestamp_column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        date_list = pd.date_range(args.starting_date, periods=args.forecast_horizon, freq=args.date_freq)

        prediction_df = pd.DataFrame()
        prediction_df['Date'] = date_list 
        prediction_df['Week_Day'] = prediction_df['Date'].apply(lambda x: x.weekday())                             

        # 4.0 Unpickle model and make predictions
        ws = Run.get_context().experiment.workspace
        models = Model.list(ws, tags=[['Store', store_name], ['Brand', brand_name], ['ModelType', 'lr']], latest=True)
        if len(models) > 1:
            raise ValueError("More than one models encountered for store and brand")
        model_path = models[0].download()
        model = joblib.load(model_path)

        
        prediction_list = []
        i = 0
        for date in prediction_df['Date']:
       
            x_pred = prediction_df[prediction_df['Date'] == date]
    
        
            if i >= 1: 
                x_pred['lag_1'] = prediction_list[i-1]
            else: 
                x_pred['lag_1'] = data[args.target_column].iloc[-1]
            if i >= 2: 
                x_pred['lag_2'] = prediction_list[i-2]
            else: 
                x_pred['lag_2'] = data[args.target_column].iloc[-2+i]
            if i >= 3: 
                x_pred['lag_3'] = prediction_list[i-3]
            else: 
                x_pred['lag_3'] = data[args.target_column].iloc[-3+i]

            y_pred = model.predict(x_pred.drop(columns=['Date']))


            prediction_list.append(y_pred[0])
            i += 1

        prediction_df['Prediction'] = prediction_list
        prediction_df['Store'] = store_name
        prediction_df['Brand'] = brand_name
        prediction_df.drop(columns=['Week_Day'], inplace=True)
        

        results = results.append(prediction_df)

    return results
