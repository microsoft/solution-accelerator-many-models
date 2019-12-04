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

thisrun = Run.get_context()
#childrun=thisrun

LOG_NAME = "user_log"

print("Make predictions")

parser = argparse.ArgumentParser("split")
parser.add_argument("--n_predictions", type=int, help="input number of predictions")

args, unknown = parser.parse_known_args()
# args = parser.parse_args()

print("Argument 1(n_predictions): %s" % args.n_predictions)

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
    print(model)
    print(model[0])
    
    for model in model:
        u1 = uuid.uuid4()
#         mname='arima'+str(input_data)
        mname='arima'+str(u1)[0:16]
        print(model)
        #for w in range(0,1):
        with thisrun.child_run(name=mname) as childrun:
            for w in range(0,5):
                thisrun.log(mname,str(w))
            date1=datetime.datetime.now()
            logger.info('starting ('+model+') ' + str(date1))
            childrun.log(mname,'starttime-'+str(date1))
            
            # 0. unpickle model 
            model_path = Model.get_model_path(model.name)
            print(model.name)
            print(model_path)
            model = joblib.load(model_path)
            print("UNPICKELED THE MODEL")
            # 1. make preidtions 
            predictions, conf_int = model.predict(args.n_predictions, return_conf_int = True)
            print("MADE PREDICTIONS")
            print(predictions)
            
            # 2. Save the output 
            #pred_start_date = # this should be the last data of the training set.. 
            
            # 3.train the model
           
            # 4. save the model
            #logger.info(model)
            #logger.info(mname)
            #with open(mname, 'wb') as file:
            #    joblib.dump(value=model, filename=os.path.join('./outputs/', mname))
            # 5. Register the model
            #ws1 = childrun.experiment.workspace
            #try:
            #    childrun.upload_file(mname, os.path.join('./outputs/', mname))
            #except:
            #    logger.info('dont need to upload')
            #logger.info('register model, skip the outputs prefix')
            #Model.register(workspace=ws1, model_path=os.path.join('./outputs/', mname), model_name='arima_'+str(input_data).split('/')[-1][:-6], model_framework='pmdarima')
            
            #you can return anything you want
            date2=datetime.datetime.now()
            logger.info('ending ('+str(file)+') ' + str(date2))

            #log some metrics
            childrun.log(mname,'endtime-'+str(date2))
            childrun.log(mname,'auc-1')
        resultList.append(True)
    return resultList