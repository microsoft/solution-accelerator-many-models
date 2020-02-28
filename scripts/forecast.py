# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import argparse
from sklearn.externals import joblib
from azureml.core.model import Model
from entry_script import EntryScript

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forecasting") #change this to tak the last date and start from there
parser.add_argument("--model_type", type=str, help="model type")
parser.add_argument("--date_freq", type=str, help="the step size for predictions, daily, weekly")

args, _ = parser.parse_known_args()


def run(input_data):
    # 1.0 Set up Logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Making forecasts')
    os.makedirs('./outputs', exist_ok=True)

    results = pd.DataFrame()
    current_run = Run.get_context()

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        date1 = datetime.datetime.now()
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = args.model_type + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]
        
        logger.info('starting ('+csv_file_path+') ' + str(date1))
       
        # 3.0 Set up data to predict on
        
        date_list = pd.date_range(args.starting_date, periods=args.forecast_horizon, freq=args.date_freq)
        predictions = pd.DataFrame()

        # 4.0 Unpickle model and make predictions
        model_path = Model.get_model_path(model_name)
        model = joblib.load(model_path)

        prediction_list = model.predict(args.forecast_horizon)
        
        predictions['date'] = date_list
        predictions['store'] = store_name
        prediciotns['brand'] = brand_name
        predictions['Predictions'] = prediction_list

        results = results.append(predictions)
        
        # Save the forecast output as individual files back to blob storage (optional)
        '''If you'd like to upload individual predictioon files as opposed to a concatenated prediction file,
        uncomment the following code block.'''
        #run_date = datetime.datetime.now().date()
        #ws = current_run.experiment.workspace
        #output_path = os.path.join('./outputs/', model_name + str(run_date))
        #prediction_df.to_csv(path_or_buf=output_path + '.csv', index = False)
        #forecasting_dstore = Datastore(ws, args.output_datastore)
        #forecasting_dstore.upload_files([output_path + '.csv'], target_path='oj_forecasts' + str(run_date),
        #                                overwrite=args.overwrite_forecasting, show_progress=True)

        # 5.0 Log the run 
        date2 = datetime.datetime.now()
        logger.info('ending (' + str(csv_file_path) +') ' + str(date2))

    return results
