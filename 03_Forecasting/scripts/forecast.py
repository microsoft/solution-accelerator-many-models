import pandas as pd
import os
import argparse
from sklearn.externals import joblib
from azureml.core.model import Model
from azureml.core.run import Run
from entry_script import EntryScript

# 0.0 Parse input arguments
parser = argparse.ArgumentParser("split")
parser.add_argument("--forecast_horizon", type=int, help="input number of predictions")
parser.add_argument("--starting_date", type=str, help="date to begin forecasting")
parser.add_argument("--output-dir", type=str, help="output directory")

args, _ = parser.parse_known_args()


def run(input_data):
    current_run = Run.get_context()
    # 1.0 Set up Logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Making forecasts')
    all_predictions = pd.DataFrame()

    # 2.0 Iterate through input data
    for idx, csv_file_path in enumerate(input_data):
        file_name = os.path.basename(csv_file_path)[:-4]
        model_name = 'arima_' + file_name
        store_name = file_name.split('_')[0]
        brand_name = file_name.split('_')[1]

        # 3.0 Set up data to predict on
        store_list = [store_name] * args.forecast_horizon
        brand_list = [brand_name] * args.forecast_horizon
        date_list = pd.date_range(args.starting_date, periods = args.forecast_horizon, freq ='W-THU')

        prediction_df = pd.DataFrame(list(zip(date_list, store_list, brand_list)),
                                    columns=['WeekStarting', 'Store', 'Brand'])

        # 4.0 Unpickle model and make predictions
        model_path = Model.get_model_path(model_name)
        model = joblib.load(model_path)

        prediction_list, conf_int = model.predict(args.forecast_horizon, return_conf_int = True)
        prediction_df['Predictions'] = prediction_list
        all_predictions = all_predictions.append(prediction_df)

        output_path = os.path.join(args.output_dir, model_name + '.csv')
        prediction_df.to_csv(output_path, index = False)

    return all_predictions
