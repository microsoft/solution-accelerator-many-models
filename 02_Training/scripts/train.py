
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
from entry_script_helper import EntryScriptHelper
import logging

from sklearn.externals import joblib
from joblib import dump, load
import pmdarima as pm
import time
from datetime import timedelta

thisrun = Run.get_context()

LOG_NAME = "user_log"

print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input target column")
parser.add_argument("--n_test_periods", type=int, help="input number of test periods")
parser.add_argument("--timestamp_column", type=str, help="input timestamp column")

args, unknown = parser.parse_known_args()

print("Argument 1(n_test_periods): %s" % args.n_test_periods)
print("Argument 2(target_column): %s" % args.target_column)
print("Argument 3(timestamp_column): %s" % args.timestamp_column)

def init():
    EntryScriptHelper().config(LOG_NAME)
    logger = logging.getLogger(LOG_NAME)
    output_folder = os.path.join(os.environ.get("AZ_BATCHAI_INPUT_AZUREML", ""), "temp/output")
    logger.info(f"{__file__}.output_folder:{output_folder}")
    logger.info("init()")
    return

def run(input_data):
    logger = logging.getLogger(LOG_NAME)
    os.makedirs('./outputs', exist_ok=True)
    resultList = []
    logger.info('processing all files')
    for file in input_data:
        u1 = uuid.uuid4()
        mname='arima'+str(u1)[0:16]
        with thisrun.child_run(name=mname) as childrun:
            for w in range(0,5):
                thisrun.log(mname,str(w))
            date1=datetime.datetime.now()
            logger.info('starting ('+file+') ' + str(date1))
            childrun.log(mname,'starttime-'+str(date1))
            # 1. Read in the data file
            data = pd.read_csv(file,header=0)
            logger.info(data.head())
            # 2. Split the data into train and test sets
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
                      #d=None,
                      seasonal=True,
                      #trend = None, # adjust this if the series have trend
                      #start_P=0, 
                      #D=0,
                      information_criterion = 'aic',
                      trace=True, #prints status on the fits 
                      #error_action='ignore', 
                      stepwise = False, # this increments instead of doing a grid search
                      suppress_warnings = True, 
                      out_of_sample_size = 16
                     )
            model = model.fit(train[args.target_column])
            logger.info('done training')
            # 4. Save the model
            logger.info(model)
            logger.info(mname)
            with open(mname, 'wb') as file:
                joblib.dump(value=model, filename=os.path.join('./outputs/', mname))
            # 5. Register the model
            ws1 = childrun.experiment.workspace
            try:
                childrun.upload_file(mname, os.path.join('./outputs/', mname))
            except:
                logger.info('dont need to upload')
            logger.info('register model, skip the outputs prefix')
            Model.register(workspace=ws1, model_path=os.path.join('./outputs/', mname), model_name='arima_'+str(input_data).split('/')[-1][:-6], model_framework='pmdarima')
            date2=datetime.datetime.now()
            logger.info('ending ('+str(file)+') ' + str(date2))
            #6. Log some metrics
            childrun.log(mname,'endtime-'+str(date2))
            childrun.log(mname,'auc-1')
        resultList.append(True)
    return resultList
