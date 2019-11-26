import pandas as pd
import os
import uuid

from azureml.core.model import Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
from azureml.core import Experiment, Workspace, Run
from azureml.core import ScriptRunConfig
import datetime
from entry_script_helper import EntryScriptHelper
import logging

from sklearn.externals import joblib
from joblib import dump, load

thisrun = Run.get_context()
#childrun=thisrun

LOG_NAME = "user_log"

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
        mname='sklearn_model_'+str(u1)[0:16]
        #for w in range(0,1):
        with thisrun.child_run(name=mname) as childrun:
            for w in range(0,5):
                thisrun.log(mname,str(w))
            date1=datetime.datetime.now()
            logger.info('starting ('+file+') ' + str(date1))
            childrun.log(mname,'starttime-'+str(date1))
            # 1.read in the data file
            data = pd.read_csv(file,header=None)
            logger.info(data.head())
            # 2. set the data up for training
            train,test=train_test_split(data,test_size=0.3)
            train_X=train[train.columns[0:6]]
            train_y=train[train.columns[-1]]
            test_X=test[train.columns[0:6]]
            test_y=train[train.columns[-1]]
            # 3. train a decision tree
            dtmodel=LinearRegression()
            dtmodel.fit(train_X,train_y)
            logger.info('done training')
            # 4. save the model
            logger.info(dtmodel)
            logger.info(mname)
            with open(mname, 'wb') as file:
                joblib.dump(value=dtmodel, filename=os.path.join('./outputs/', mname))
            # 5. Register the model
            ws1 = childrun.experiment.workspace
            try:
                childrun.upload_file(mname, os.path.join('./outputs/', mname))
            except:
                logger.info('dont need to upload')
            logger.info('register model, skip the outputs prefix')
            Model.register(workspace=ws1, model_path=os.path.join('./outputs/', mname), model_name=mname, model_framework='sklearn')
            #you can return anything you want
            date2=datetime.datetime.now()
            logger.info('ending ('+str(file)+') ' + str(date2))

            #log some metrics
            childrun.log(mname,'endtime-'+str(date2))
            childrun.log(mname,'auc-1')
        resultList.append(True)
    return resultList
