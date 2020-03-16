# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Run
import pandas as pd
import numpy as np
import os
import argparse
import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression 

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--forecast_granularity", type=int, help="frequency of forecasts daily weekly")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--model_type", type=str, help="input model type")

args, _ = parser.parse_known_args()

current_run = None

def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # 2.0 Loop through each file in the batch
        # The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(input_data):
        result = {}
        start_datetime = datetime.datetime.now()
        
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + '_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        data = pd.read_csv(csv_file_path, header = 0)

        # 3.0 Create Features 
        data['Week_Day'] = data[args.timestamp_column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').weekday())
          
        for i in range(1,4):
            data['lag_'+ str(i)] = data[args.target_column].shift(i)
        
        data = data.drop(['Price', 'Revenue', 'Store', 'Brand', 'Advert'], axis = 1) 
        data = data.dropna()
        print(data)


        # 4.0 Split the data into train and test sets based on dates
        data = data.set_index(args.timestamp_column)
        max_date = datetime.datetime.strptime(data.index.max(), '%Y-%m-%d')
        split_date = max_date - timedelta(days= args.forecast_granularity * args.n_test_periods)
        data.index = pd.to_datetime(data.index)
        train = data[data.index <= split_date]
        test = data[data.index > split_date]
        
        y_train = train[args.target_column]
        y_test = test[args.target_column]
        X_train = train.drop(args.target_column, axis = 1)
        X_test = test.drop(args.target_column, axis = 1)
        

        # 5.0 Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6.0 Save the model
        joblib.dump(model, filename=os.path.join('./outputs/', model_name))

        # 7.0 Register the model to the workspace
        current_run.upload_file(model_name, os.path.join('./outputs/', model_name))
                    
        tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType': args.model_type}
        current_run.register_model(model_path = model_name, model_name = model_name, model_framework = args.model_type, tags = tags_dict)

        # 8.0 Make predictions on test set
        test['Predictions'] = model.predict(X_test)
        
        # 9.0 Calculate accuracy metrics
        mse = mean_squared_error(test['Quantity'], test['Predictions'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['Quantity'], test['Predictions'])
        act, pred = np.array(test['Quantity']), np.array(test['Predictions'])
        mape = np.mean(np.abs((act - pred) / act) * 100)

        # 10.0 Log metrics
        current_run.log(model_name + '_mse', mse)
        current_run.log(model_name + '_rmse', rmse)
        current_run.log(model_name + '_mae', mae)
        current_run.log(model_name + '_mape', mape)                
                    
        # 11.0 Add data to output
        end_datetime = datetime.datetime.now()
        result['store'] = store_name
        result['brand'] = brand_name
        result['model_type'] = args.model_type
        result['file_name'] = file_name
        result['model_name'] = model_name
        result['start_date'] = str(start_datetime)
        result['end_date'] = str(end_datetime)
        result['duration'] = str(end_datetime-start_datetime)
        result['mse'] = mse
        result['rmse'] = rmse
        result['mae'] = mae
        result['mape'] = mape
        result['index'] = idx
        result['num_models'] = len(input_data)
        result['status'] = current_run.get_status()

        print('ending (' + csv_file_path + ') ' + str(end_datetime))
        result_list.append(result)

    # Data returned by this function will be available in parallel_run_step.txt
    return pd.DataFrame(result_list)