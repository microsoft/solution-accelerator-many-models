
from azureml.core import Experiment, Workspace, Run
import pmdarima as pm
import pandas as pd
import os
import logging
import argparse
import datetime
from datetime import timedelta
from sklearn.externals import joblib
from joblib import dump, load
from entry_script_helper import EntryScriptHelper

current_run = Run.get_context()

LOG_NAME = "user_log"

print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")
parser.add_argument("--stepwise_training", type=str, help="input stepwise training True or False")

args, unknown = parser.parse_known_args()

print("Argument 1(n_test_periods): %s" % args.n_test_periods)
print("Argument 2(target_column): %s" % args.target_column)
print("Argument 3(timestamp_column): %s" % args.timestamp_column)
print("Argument 4(stepwise_training): %s" % args.stepwise_training)

def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")
    return


def run(input_data):
    # 0. Set up logging
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    logger.info('processing all files')
    resultList = []

    # 1. Read in the data file
    for idx, csv_file_path in enumerate(input_data):
        logs = []
        date1=datetime.datetime.now()
        logger.info('starting ('+csv_file_path+') ' + str(date1))
        data = pd.read_csv(csv_file_path,header=0)
        logger.info(data.head())
        model_name = 'arima_'+str(input_data).split('/')[-1][:-6]

        # 2. Split the data into train and test sets based on dates
        try:
            data = data.set_index(args.timestamp_column)
            max_date = datetime.datetime.strptime(data.index.max(),'%Y-%m-%d')
            split_date = max_date - timedelta(days=7*args.n_test_periods)
            data.index = pd.to_datetime(data.index)
            train = data[data.index <= split_date]
            test = data[data.index > split_date]

        # 3.Train the model
            model = pm.auto_arima(train[args.target_column],
                  start_p=0,
                  start_q=0,
                  test='adf', #default stationarity test is kpps
                  max_p =3,
                  max_d = 2,
                  max_q=3,
                  m=3, #number of observations per seasonal cycle
                  seasonal=True,
                  information_criterion = 'aic',
                  trace=True, #prints status on the fits
                  stepwise = args.stepwise_training, # this increments instead of doing a grid search
                  suppress_warnings = True,
                  out_of_sample_size = 16
                 )
            model = model.fit(train[args.target_column])
            logger.info('done training')

            # 4. Save the model
            logger.info(model)
            with open(model_name, 'wb') as file:
                joblib.dump(value=model, filename=os.path.join('./outputs/', model_name))

            # 5. Register the model to the workspace
            ws1 = current_run.experiment.workspace
            try:
                current_run.upload_file(model_name, os.path.join('./outputs/', model_name))
            except:
                logger.info('dont need to upload')
            logger.info('register model, skip the outputs prefix')
            print('Trained '+ model_name)

            tags_dict={'Store': str(csv_file_path).split('/')[-1][:-4].split('_')[0], 'Brand': str(csv_file_path).split('/')[-1][:-4].split('_')[1], 'ModelType':'ARIMA'}
            current_run.register_model(model_path=model_name, model_name=model_name, model_framework='pmdarima',tags=tags_dict)
            print('Registered '+ model_name)

            #6. Log some metrics
            date2=datetime.datetime.now()
            logger.info('ending ('+str(csv_file_path)+') ' + str(date2))

            logs.append(str(csv_file_path).split('/')[-1][:-4].split('_')[0])
            logs.append(str(csv_file_path).split('/')[-1][:-4].split('_')[1])
            logs.append('ARIMA')
            logs.append(str(csv_file_path).split('/')[-1][:-4])
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2-date1))
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(current_run.get_status())
            AIC = 'AIC ' + str(model.aic())
            current_run.log(model_name, AIC)
        except Exception as e:
            model_name = 'arima_'+str(input_data).split('/')[-1][:-6]
            date2=datetime.datetime.now()
            error_message = 'Failed to train the model. '+'Error message: '+str(e)

            logs.append(str(csv_file_path).split('/')[-1][:-4].split('_')[0])
            logs.append(str(csv_file_path).split('/')[-1][:-4].split('_')[1])
            logs.append('ARIMA')
            logs.append(str(csv_file_path).split('/')[-1][:-4])
            logs.append(model_name)
            logs.append(str(date1))
            logs.append(str(date2))
            logs.append(str(date2-date1))
            logs.append(idx)
            logs.append(len(input_data))
            logs.append(error_message)

    resultList.append(logs)
    return resultList
