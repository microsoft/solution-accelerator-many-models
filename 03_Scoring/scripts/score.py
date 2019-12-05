from azureml.core.run import Run
import pandas as pd
import os
import uuid
import argparse
import datetime

from azureml.core.model import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig
# import datetime
from entry_script_helper import EntryScriptHelper
import logging

from sklearn.externals import joblib
from joblib import dump, load
import pmdarima as pm
import time
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error 

thisrun = Run.get_context()
#childrun=thisrun

LOG_NAME = "user_log"

print("Make predictions")

parser = argparse.ArgumentParser("split")
parser.add_argument("--n_predictions", type=int, help="input number of predictions")
parser.add_argument("--model", type=str, help="model name")
#parser.add_argument("--start_date", type=str, help="date to start predictions")

args, unknown = parser.parse_known_args()
# args = parser.parse_args()

print("Argument 1(n_predictions): %s" % args.n_predictions)
print("Argument 2(model): %s" % args.model)

def mape_calc(actual, predicted):
    act, pred = np.array(actual), np.array(predicted)
    mape = np.mean(np.abs((act - pred)/act)*100)
    return mape

def get_accuracy_metrics(actual, predicted, print_values = True):

    metrics = []
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mape_calc(actual, predicted)
    
    metrics.append(mse)
    metrics.append(rmse)
    metrics.append(mae)
    metrics.append(mape)
    
    if print_values == True: 
        print('Accuracy Metrics')
        print('MSE: {}'.format(mse))
        print('RMSE: {}'.format(rmse))
        print('MAE: {}'.format(mae))
        print('MAPE: {}'.format(mape))
    
    return metrics


def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")    
    return

def run(data):
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    logger.info('making predictions...')
    print("ITERATING THROUGH MODELS")
    
    for file in data:
        u1 = uuid.uuid4()
        mname='arima'+str(u1)[0:16]

        #for w in range(0,1):
        with thisrun.child_run(name=mname) as childrun:
            for w in range(0,5):
                thisrun.log(mname,str(w))
            date1=datetime.datetime.now()
            logger.info('starting ('+file+') ' + str(date1))
            childrun.log(mname,'starttime-'+str(date1))
            
            # 0. unpickle model 
            model_path = Model.get_model_path(args.model) # can we parse the name of the csv for the model name?
            print(model_path)
            model = joblib.load(model_path)
            print("UNPICKELED THE MODEL")
            # 1. make preidtions 
            predictions, conf_int = model.predict(args.n_predictions, return_conf_int = True)
            print("MADE PREDICTIONS")
            print(predictions)
            
            # 2. Score predictions with test set 
            test = pd.read_csv(file,header=0, )
            logger.info(data.head())
             
            test['Predicitons'] = predictions
            
            # accuracy metrics 
            accuracy_metrics = get_accuracy_metrics(test['Quantity'], test['Predictions'])
            print(accuracy_metrics)
            logger.info(accuracy_metrics)
            
            # 3. Save the output back to blob storage 
            predictions_path = 'predictions'
            filename = '/arima_'+str(input_data).split('/')[-1][:-6]+'.csv'
            
            test[['Quantity', 'Predictions']].to_csv(path_or_buf = predictions_path + filename, index = False)
           
            #you can return anything you want
            date2=datetime.datetime.now()
            logger.info('ending ('+str(file)+') ' + str(date2))

            #log some metrics
            childrun.log(mname,'endtime-'+str(date2))
            childrun.log(mname,'auc-1')
        resultList.append(True)
    return resultList