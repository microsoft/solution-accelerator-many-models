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
from entry_script import EntryScript
from sklearn import LinearRegression 

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--model_type", type=str, help="input model type")

args, _ = parser.parse_known_args()

current_run = None

def init():
    global current_run
    current_run = Run.get_context()


def run(input_data):
    # 1.0 Set up logging
    entry_script = EntryScript()
    logger = entry_script.logger
    os.makedirs('./outputs', exist_ok=True)
    logger.info('Train and Score models')
    resultList = []
    log_list = []

    # 2.0 Read in the data file
    for idx, csv_file_path in enumerate(input_data):
        logs = []
        date1 = datetime.datetime.now()
        logger.info('starting ('+csv_file_path+') '+ str(date1)
        
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        data = pd.read_csv(csv_file_path, header = 0)
        logger.info(data.head())

        # 3.0 Split the data into train and test sets based on dates
        try: 
            data = data.set_index(args.timestamp_column)
            max_date = datetime.datetime.strptime(data.index.max(), '%Y-%m-%d')
            split_date = max_date - timedelta(days=7 * args.n_test_periods)
            data.index = pd.to_datetime(data.index)
            train = data[data.index <= split_date]
            test = data[data.index > split_date]
            
            y_train = train['target']
            y_test = test['target']
            X_train = train.drop('target', axis = 1)
            x_test = test.drop('target', axis = 1)

           # 4.0 Train the model
            model = LinearRegression().fit(X_train, y_train)
 
            logger.info('done training')
            print('Trained ' + model_name)

            # 5.0 Save the model
            logger.info('done training')
            joblib.dump(model, filename=os.path.join('./outputs/', model_name))

            # 6.0 Register the model to the workspace
            try: 
                current_run.upload_file(model_name, os.path.join('./outputs/', model_name))
            except:
                logger.info('dont need to upload model')
            logger.info('register model, skip the outputs prefix')
                        
            tags_dict = {'Store': store_name, 'Brand': brand_name, 'ModelType': args.model_type}
            current_run.register_model(model_path = model_name, model_name = model_name, model_framework = args.model_type, tags = tags_dict)

            # 7.0 Make predictions on test set
            prediction_list = model.predict(X_test)
        
            # 8.0 Insert prediciotns to test set 
            test['Predictions'] = prediction_list
            print(test.head())
            print('Predictions added ' + model_name)

            # 9.0 Calculate accuracy metrics
            metrics = []
            mse = mean_squared_error(test[args.target_column], test['Predictions'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test[args.target_column], test['Predictions'])
            act, pred = np.array(test[args.target_column]), np.array(test['Predictions'])
            mape = np.mean(np.abs((act - pred) / act) * 100)
            metrics.append(mse)
            metrics.append(rmse)
            metrics.append(mae)
            metrics.append(mape)
            print('Calculated accuracy metrics  ' + model_name)
            print(metrics)
                    
            # 9.1 Log metrics
            logger.info('accuracy metrics')
            logger.info(metrics) 
                        
            # 10.0 Log the run 
            date2 = datetime.datetime.now()
                logs.append(store_name)
                logs.append(brand_name)
                logs.append(args.model_type)
                logs.append(file_name)
                logs.append(model_name)
                logs.append(str(date1))
                logs.append(str(date2))
                logs.append(str(date2-date1))
                logs.append(mse)
                logs.append(rmse)
                logs.append(mae)
                logs.append(mape)
                logs.append(idx)
                logs.append(len(input_data))
                logs.append(current_run.get_status())

                log_list.append(logs)
            logger.info('ending (' + csv_file_path + ') ' + str(date2))

        # 10.1 Log error if occurred 
        except (ValueError, UnboundLocalError, NameError, ModuleNotFoundError, AttributeError, ImportError, FileNotFoundError, KeyError) as error:
            date2 = datetime.datetime.now()
            error_message = 'Failed to score the model. ' + 'Error message: ' + str(error)

            logs.append(store_name)
            logs.append(brand_name)
            logs.append('ARIMA')
            logs.append(file_name)
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2-date1))
            logs.append('Null')
            logs.append('Null')
            logs.append('Null')
            logs.append('Null')
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(error_message)

            logger.info('ending ('+csv_file_path+') ' + str(date2))
            log_list.append(logs)

    resultList.extend(log_list)
    return resultList
