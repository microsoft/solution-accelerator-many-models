import argparse
import os
from datetime import datetime 
from azureml.core import Run 
from azureml.core.model import Model
import pandas as pd 
import pmdarima as pm


def train_one_model():
    run = Run.get_context()
    LOG_NAME = "user_log"
    
    start_time = datetime.now()
    run.log("scriptStartTime", str(start_time)) 
    ###### NEED TO ADD THE INPUTS FROM THE PARALLEL RUN CONFIG HERE ######
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type = str, dest = 'var_name')
    # need to add in the store, brand, n_test_period 
    
    blob_container = args.input_blob
    
    ## IN THE DATA_PREP CODE we should set the timestamp to the index
    train_auto_arima(input_data, target_column, timestamp_column, n_test_periods)
    
    run.log("scriptRunSeconds", (datetime.now() - start_time).total_seconds())
    
def train_auto_arima(input_data, target_column, timestamp_column, n_test_periods =16):
    """
    Trains an auto_arima model on one series and adds it to the workspace model registry. 
    
    Parameters 
    ----------
    series_name: str
        name of the series
    start_date: str
        Training start date
    end_date: str
        Training end date
    container: str
        path on the blob with the input data for specified asset
    run: azureml.core.Run
        AzureML Run object
 """
    # Create Directory under outputs for model objects 
	# update directory to blob 
    model_outputs = os.path.join("outputs")
    if not os.path.exists(model_outputs):
        os.makedirs(model_outputs)
    
    logger = logging.getLogger(LOG_NAME)
    logger.info('processing all files')
    
    for file in input_data: 
        model_name = 'auto_arima_'+str(file) #parse name from input data name 
        
        with thisrun.child_run(name=model_name) as childrun: 
            start_date = datetime.now()
            logger.info('starting ('+file+') '+ str(start_date))
            
            # read in the data 
            data = pd.read_csv(file)
            logger.info(data.head())
            data = data.set_index(timestamp_column)
            
            # Split the data into train and test sets 
            n_test_periods = n_test_periods

            split_date = data.index.max() - timedelta(days=7*n_test_periods)
            train = data[data.index <= split_date]
            test = data[data.index > split_date]
            
            # Train auto_arima 
            model = pm.auto_arima(train[target_column], 
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
            logger.info('Training Complete')
            
            #### TEST and any OUTPUTS from training - MSE? ####
            
            # Save the model
            logger.info(model)
            logger.info(model_name)
            with open(model_name, 'wb') as file:
                joblib.dump(value=model, filename=os.path.join('.outputs/', model_name))
            
            # Register the Model 
            ws_child = childrun.experiment.workspace
            try: 
                childrun.upload_file(model_name, os.path.join('.outputs/', model_name))
            except: 
                logger.info('Do not need to upload')
            logger.info('Register Model')
            
            Model.register(workspace = ws_child, model_path = os.path.join('.outputs/', model_name), model_name = model_name, 
                          model_framework = 'auto_arima')
            end_date = datetime.now()
            logger.info('Completing ('+str(file)+') '+ str(end_date))
            
            
if __name__ == "__main__":
    train_one_model()
    