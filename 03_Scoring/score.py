import pickle
import json
import numpy
import argparse
import os
from sklearn.externals import joblib
from azureml.core.model import Model
from azureml.core import Workspace, Dataset
from azureml.core.run import Run, _OfflineRun
import azureml.train.automl

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str)
parser.add_argument("--dataset_to_score", type=str)
parser.add_argument("--output_scores", type=str, help="output file for scores")
args = parser.parse_args()

run = Run.get_context()
ws = None
if type(run)==_OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace
    
score_ds = Dataset.get(ws, args.dataset_to_score)
score_df = score_ds.to_pandas_dataframe()
data = score_df.drop(["ACCOUNT_NUMBER"], axis=1)
cc = [i for i in range(0, len(data.columns))]
data.columns = cc

print(data.columns)
print(data.head(3))

model_path = Model.get_model_path(args.model_id)

print(model_path)

model = joblib.load(model_path)
result = model.predict(data)

score_df['prediction'] = result
score_df['model_id'] = args.model_id 

outputs = score_df[['ACCOUNT_NUMBER', 'model_id', 'prediction']]


print(outputs.head(3))

os.makedirs(args.output_scores, exist_ok=True)
outputs.to_csv(args.output_scores + "predictions.csv", index=False)

